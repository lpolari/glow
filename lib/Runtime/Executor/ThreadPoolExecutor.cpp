/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ExecutionState.h"

#include "glow/Backends/DeviceManager.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Runtime/Executor/ThreadPoolExecutor.h"

#include <queue>
#include <unordered_set>

#include "llvm/Support/FormatVariadic.h"
#include <glog/logging.h>

namespace glow {
namespace runtime {

std::mutex network_mutex[10];

void InflightBarrier::decrement(unsigned decr) {
  std::unique_lock<std::mutex> lock(mtx_);
  DCHECK_GE(count_, decr) << "Barrier decrement cannot be less than count!";
  count_ -= decr;

  // If count_ has hit zero, wake up all threads that are waiting.
  if (count_ == 0) {
    cv_.notify_all();
  }
} // namespace runtime

void InflightBarrier::increment(unsigned incr) {
  std::unique_lock<std::mutex> lock(mtx_);
  count_ += incr;
}

unsigned InflightBarrier::count() {
  std::unique_lock<std::mutex> lock(mtx_);
  return count_;
}

void InflightBarrier::wait() {
  std::unique_lock<std::mutex> lock(mtx_);
  // If count_ is not 0, wait until a signal is received that it is.
  // The second argument below is a predicate that returns true when
  // it is safe to wake up. It preserves correctness in the case of
  // spurious wakeups.
  cv_.wait(lock, [&] { return count_ == 0; });
}

void TimeslotBarrier::decrement(unsigned decr) {
  std::unique_lock<std::mutex> lock(mtx_);
  DCHECK_GE(count_, decr) << "Barrier decrement cannot be less than count!";
  count_ -= decr;

  cv_.notify_all();
}

void TimeslotBarrier::increment(unsigned incr) {
  std::unique_lock<std::mutex> lock(mtx_);
  count_ += incr;
}

unsigned TimeslotBarrier::count() {
  std::unique_lock<std::mutex> lock(mtx_);
  return count_;
}

void TimeslotBarrier::wait(unsigned level) {
  std::unique_lock<std::mutex> lock(mtx_);
  // If count_ is not 0, wait until a signal is received that it is.
  // The second argument below is a predicate that returns true when
  // it is safe to wake up. It preserves correctness in the case of
  // spurious wakeups.
  cv_.wait(lock, [&] { return count_ <= level; });
}

void TimeslotBarrier::setFramenumber(unsigned framenumber){
  this->framenumber = framenumber;
}

unsigned TimeslotBarrier::getFramenumber(){
  return this->framenumber;
}

ThreadPoolExecutor::ThreadPoolExecutor(const DeviceManagerMapTy
                                           &deviceManagers,
                                       unsigned numWorkers,
                                       const std::string &name,
                                       const std::vector<TimeslotBarrier*>
                                           *timeslotBarriers)
    : threadPool_(numWorkers,
                  std::make_shared<folly::NamedThreadFactory>(name)),
      deviceManagers_(deviceManagers){
        this->timeslotBarriers_ = timeslotBarriers;
}

void ThreadPoolExecutor::shutdown() {
  // Prevent more requests from being processed.
  shuttingDown_ = true;

  // Wait for all inflight DeviceManager::runFunction() calls to return and be
  // processed before starting to destroy state that is used in
  // handleDeviceManagerResult().
  inflightBarrier_.wait();

  threadPool_.stop();
  threadPool_.join();
}

void publishTimeslotBarriersToExecutionState(NetworkExecutionState *state,
                                          const DAGNode *node,
    unsigned offset, unsigned period, unsigned level){
  state->setNodeOffset(node, (offset + level) % 4);
  for (DAGNode *child : node->children){
    publishTimeslotBarriersToExecutionState(state, child , offset,
                                            period, ++level);
  }
}


void ThreadPoolExecutor::run(const DAGNode *root,
                             std::unique_ptr<ExecutionContext> context,
                             RunIdentifierTy runId, ResultCBTy cb) {
  DCHECK(cb != nullptr);

  unsigned criticality = context->getCriticality();
  time_point<clock_type, milliseconds> deadline = context->getNextDeadline();

  TRACE_EVENT_SCOPE(context->getTraceContext(), TraceLevel::RUNTIME,
                    "ThreadPoolExecutor::run");

  if (context->getTraceContext()) {
    auto tid = threads::getThreadId();
    if (!context->getTraceContext()->getThreadNames().count(tid)) {
      context->getTraceContext()->setThreadName(tid, "ThreadPoolExecutor");
    }
  }

  // Don't process new requests if the executor is shutting down.
  if (shuttingDown_) {
    cb(runId,
       MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_REQUEST_REFUSED,
                "ThreadPoolExecutor is shutting down"),
       std::move(context));
    return;
  }

  // If list of roots is empty, there is nothing to do. Give back the
  // bindings so the caller can reuse it.
  if (!root) {
    cb(runId, Error::success(), std::move(context));
    return;
  }

  auto numChildren = (root->children).size();
  // Mark the child nodes as "inflight" (i.e. currently executing). This must
  // be done here instead of inside executeDAGNode() so that a node can be
  // executed while placeholders are being propagated for the next node
  // without the callback for that node deleting the execution state.
  inflightBarrier_.increment(numChildren);

  auto *traceContext = context->getTraceContext();

  // Get and bind state.
  auto currentState = states_[root]->getNextNetworkExecutionState();

  currentState->setTimeslotBarriers(this->timeslotBarriers_);
  currentState->setPeriod(context->getPeriod());

  for (const DAGNode *child : root->children){
    publishTimeslotBarriersToExecutionState(currentState, child,
        context->getOffset(), context->getPeriod(), 0);
  }

  TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME,
                    "bind network execution state");
  currentState->bind(std::move(context), std::move(cb), runId);
  TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME,
                  "bind network execution state");

  currentState->incrementInflightNodes(numChildren);

  // End the trace block before calling executeDAGNode() which can trigger the
  // result cb. Once the result cb is called, it's no longer safe to access the
  // trace context.
  TRACE_EVENT_SCOPE_END();
  for (auto const &node : root->children) {
    // Run with cached state
    executeDAGNodeAsync(currentState, node, criticality, deadline);
  }
}

void ThreadPoolExecutor::executeDAGNodeAsync(NetworkExecutionState *executionState,
                                             DAGNode *node, unsigned criticality,
                                             time_point
                                                 <clock_type,
                                              milliseconds> deadline){
  threadPool_.add([this, executionState, node, criticality, deadline]() mutable {
    executeDAGNode(executionState, node, criticality, deadline);
  });
}

void ThreadPoolExecutor::executeDAGNode(NetworkExecutionState *executionState,
                                        DAGNode *node, unsigned criticality,
                                        time_point
                                            <clock_type,
                                         milliseconds> deadline) {
  TRACE_EVENT_SCOPE(executionState->getRawResultContextPtr()->getTraceContext(),
                    TraceLevel::RUNTIME, "ThreadPoolExecutor::executeDAGNode");

  /// Set maximum scheduling priority for the executor
  auto thread_id = std::this_thread::get_id();
  auto native_handle = *reinterpret_cast<std::thread::native_handle_type*>(&thread_id);
  sched_param* param = (sched_param*) malloc(sizeof(sched_param));
  param->sched_priority = sched_get_priority_max(SCHED_FIFO);
  pthread_setschedparam(native_handle, SCHED_FIFO, param);

  /// Wait for the timeslot barrier to open
  TimeslotBarrier *timeslotBarrier = executionState
                                         ->getTimeslotBarrier(node);
  size_t offset = timeslotBarrier->getFramenumber();
  LOG(INFO) << "[" << node->name << "] Wait for frame " << offset << "!\n";
  LOG(INFO) << "[" << node->name << "] Current barrier counter :: "
            << timeslotBarrier->count() << "\n";
  timeslotBarrier->wait(criticality);
  TRACE_EVENT_BEGIN(executionState->getRawResultContextPtr()
                        ->getTraceContext(),
                    TraceLevel::RUNTIME, "Enter timeslot");

  LOG(INFO) << "[" << node->name << "] Barrier is open for frame "
            << timeslotBarrier->getFramenumber() << "\n";

  if (executionState->getErrorContainer().containsErr()) {
    // LPolariToDo uncomment error checking
    // Mark the node as no longer executing.
    //executionState->decrementInflightNodes();
    //inflightBarrier_.decrement();
  }

  // Get the PlaceholderBindings containing all of the inputs for the node.
  std::unique_ptr<ExecutionContext> nodeCtx =
      executionState->getUniqueNodeContextPtr(node);
  nodeCtx->setCriticality(criticality);
  nodeCtx->setNextDeadline(deadline);
  nodeCtx->setOffset(offset);

  // Set context name. This is the same as the module / network name
  // This is needed later to declare the global continue execution flag
  // for the functions of the module
  std::string function_name = node->name;
  int found = function_name.find_last_of("_");
  std::string module_name = function_name.substr(0,found);
  nodeCtx->setName(module_name);

  // Get the DeviceManager that can run the node.
  auto currentDevice = node->getNextDevice();
  auto deviceManagerIt = deviceManagers_.find(currentDevice);

  if (deviceManagerIt == deviceManagers_.end()) {
    // Mark the node as no longer executing.
    executionState->getErrorContainer().set(
        MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_DEVICE_NOT_FOUND,
                 "Cannot find the DeviceManager specified."));
    executionState->decrementInflightNodes();
    inflightBarrier_.decrement();
    return;
  }
  DeviceManager *deviceManager = deviceManagerIt->second.get();
  // If the context has a deviceManager bound use that instead.
  if (nodeCtx->getBoundDeviceManager()) {
    deviceManager = nodeCtx->getBoundDeviceManager();
  }

  TRACE_EVENT_END(executionState->getRawResultContextPtr()->getTraceContext(),
                  TraceLevel::RUNTIME, "Enter timeslot");

  // End the trace block before calling deviceManager->runFunction which can
  // trigger the result cb in a different thread. Once the result cb is called,
  // it's no longer safe to access the trace context.
  TRACE_EVENT_SCOPE_END();
  // Run the node using the DeviceManager.
  LOG(INFO) << "[" << node->name << "] Execute function via device manager!\n";

  deviceManager->runFunction(
      node->name, std::move(nodeCtx), timeslotBarrier,
      [this, executionState, currentDevice,
       node](RunIdentifierTy id, Error err,
             std::unique_ptr<ExecutionContext> resultCtx) {
        TRACE_EVENT_LOG_ID(resultCtx->getTraceContext(), TraceLevel::REQUEST,
                           "handle result queuing", TraceEvent::AsyncBeginType,
                           TraceEvent::now(), id);

        // Immediately move the handling of the result onto this run's executor
        // to avoid doing work on the DeviceManager thread.
        threadPool_.add([this, executionState, node, err = std::move(err),
                         currentDevice, id,
                         ctx = std::move(resultCtx)]() mutable {
          TRACE_EVENT_LOG_ID(ctx->getTraceContext(), TraceLevel::REQUEST,
                             "handle result queuing", TraceEvent::AsyncEndType,
                             TraceEvent::now(), id);

          if (!err){
            node->markFinished(currentDevice);
          }
          this->handleDeviceManagerResult(executionState, std::move(err),
                                          std::move(ctx), node);
        });
      });
}

void ThreadPoolExecutor::handleDeviceManagerResult(
    NetworkExecutionState *executionState, Error err,
    std::unique_ptr<ExecutionContext> ctx, /*const*/ DAGNode *node) {
  size_t pos = node->name.find("_id");
  std::string str2 = node->name.substr (pos+3,1);
  size_t index = stoi(str2) -  1;
  LOG(INFO) << "Lock network mutex for network id " << index;
  network_mutex[index].lock();

  TraceContext *traceContext = ctx->getTraceContext();

  LOG(INFO) << "[" << node->name << "] Handle device manager results\n";
  
  //  Decrement the timeslotBarrier for the following frame after each
  //  subnet execution
  //  New networks should only execute when all previous networks
  //  returned
  LOG(INFO) << "[" << node->name
            << "] Decrement TimeslotBarrier for following timeslot "
            << std::to_string((ctx->getOffset() + 1) % 4);
  TimeslotBarrier *timeslotBarrier = executionState->getNextTimeslotBarrier(node);
  timeslotBarrier->decrement();

  if (traceContext) {
    TRACE_EVENT_BEGIN(traceContext, TraceLevel::RUNTIME,
                      "ThreadPoolExecutor::handleResult");
  }
  auto runWasSuccess = !err;

  // Set the result code for the run.
  executionState->getErrorContainer().set(std::move(err));

  //   GLOBALPERIODMARKER
  size_t criticality = ctx->getCriticality();
  time_point<clock_type, milliseconds>
      deadline = ctx->getNextDeadline() + this->timeslotSize;
  size_t offset = (ctx->getOffset() + 1) % 4;
  LOG(INFO) << "[" << node->name << "] Increment offset to " << offset;
  LOG(INFO) << "[" << node->name << "] Increment deadline to "
            << deadline.time_since_epoch().count();

  //LOG(INFO) << "[" << node->name << "] Return intermeddiate execution state";
  executionState->returnUniqueNodeContextPtr(node, std::move(ctx));

  // If the DeviceManager executed the node, propagate its output Placeholders
  // to its children or the result PlaceholderBindings as appropriate.
  if (runWasSuccess) {
    LOG(INFO) << "[" << node->name << "] Run was successful\n";

    for (auto &child : node->children) {
      // Execute any child that has no parent nodes left to execute.
      bool childReadyToExecute =
          executionState->incrementNodeParentsDone(child);
      if (childReadyToExecute) {
        // Mark the node as "inflight" (i.e. currently executing).
        executionState->incrementInflightNodes();
        inflightBarrier_.increment();
        LOG(INFO) << "[" << node->name << "] Dispatch next\n";
        executeDAGNode(executionState, child, criticality, deadline);

        //executionState->returnUniqueNodeContextPtr(node, std::move(ctx));
        executionState->decrementInflightNodes();
        inflightBarrier_.decrement();

        if (traceContext) {
          TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME,
                          "ThreadPoolExecutor::handleResult");
          executionState->insertIntoTraceContext(traceContext);
        }

        LOG(INFO) << "Unlock network mutex for network id " << index;
        network_mutex[index].unlock();
        return;
      }
    }
  }

  // If the run was not successful, shift all subnets of the corresponding
  // dnn and execute the current one again
  if (!runWasSuccess){
    LOG(INFO) << "[" << node->name << "] Run was successful\n";
    executionState->incrementInflightNodes();
    inflightBarrier_.increment();
    executionState->shiftTimeslotBarriers();
    LOG(INFO) << "[" << node->name << "] Dispatch again\n";
    executeDAGNode(executionState, node, criticality, deadline);
    executionState->decrementInflightNodes();
    inflightBarrier_.decrement();

    if (traceContext) {
      TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME,
                      "ThreadPoolExecutor::handleResult");
      executionState->insertIntoTraceContext(traceContext);
    }

    LOG(INFO) << "Unlock network mutex for network id " << index;
    network_mutex[index].unlock();
    return;
  }

  // This needs to happen before decrementInflightNodes(). Otherwise a race
  // condition can happen where two threads call into this function at the same
  // time. Once decrementInflightNodes() is called, only the thread that get
  // noNodesInflight == true can access executionState.
  if (traceContext) {
    TRACE_EVENT_END(traceContext, TraceLevel::RUNTIME,
                    "ThreadPoolExecutor::handleResult");
    executionState->insertIntoTraceContext(traceContext);
  }

  // Now, check if all nodes in the graph are done. If so, the callback can be
  // called and all state associated with the run can be erased.
  bool noNodesInflight = executionState->decrementInflightNodes();

  if (noNodesInflight) {
    // If there are no nodes inflight, that means all nodes are done. Transfer
    // the outpus. Call the callback and erase the state information.
    // Because we are redirecting inputs and outputs to use the provided tensor
    // we do not have to transfer outputs here. Once we have pinned memory we
    // will transfer. //executionState->transferOutputs();
    ResultCBTy cb = executionState->getCallback();
    DCHECK(cb != nullptr);

    // Get what we need from the executionState and return it to the pool.
    auto runId = executionState->getRunId();
    auto err = executionState->getErrorContainer().get();
    auto resultCtx = executionState->getUniqueResultContextPtr();
    LOG(INFO) << "[" << node->name << "] Set result offset " << offset;
    resultCtx->setOffset(offset);
    resultCtx->setNextDeadline(deadline);
    states_[executionState->getRoot()]->returnNetworkExecutionState(
        executionState);

    cb(runId, std::move(err), std::move(resultCtx));
  }

  // Decrement the inflight barrier for the executor keeping track of all
  // outstanding DeviceManager::runFunction() calls. This must be done here
  // instead of right after executionState->decrementInflightNodes() so that
  // ~ThreadPoolExecutor does not delete executor state before this function
  // is done using it (e.g. when erasing the ExecutionState object for a
  // run).
  inflightBarrier_.decrement();
  LOG(INFO) << "Unlock network mutex for network id " << index;
  network_mutex[index].unlock();
}

void ThreadPoolExecutor::createPool(const DAGNode *root, unsigned poolSize,
                                    bool assignStatic) {
  std::unordered_map<DAGNode *, DeviceIDTy> assignment;

  // For static assignment we need to track devices each node is assigned to.
  std::unordered_map<DAGNode *, std::vector<DeviceIDTy>> assignments;
  std::unordered_map<DAGNode *, unsigned> currentAssignment;
  if (assignStatic) {
    // Walk the nodes and get assignments.
    std::queue<DAGNode *> remaining;
    for (auto node : root->children) {
      remaining.push(node);
    }
    while (remaining.size()) {
      auto node = remaining.front();
      remaining.pop();
      // Add any new children to the queue.
      for (auto child : node->children) {
        auto it = assignments.find(child);
        if (it == assignments.end()) {
          remaining.push(child);
        }
      }
      std::vector<DeviceIDTy> assignment;
      for (auto dev : node->deviceRuntimeInfos) {
        assignment.push_back(dev.first);
      }
      assignments[node] = assignment;
      currentAssignment[node] = 0;
    }
  }

  std::unique_ptr<NetworkExecutionStatePool> pool =
      glow::make_unique<NetworkExecutionStatePool>();
  for (unsigned i = 0; i < poolSize; i++) {
    auto newState = glow::make_unique<NetworkExecutionState>(root);
    // If assignStatic, calculate the device assignments for this
    // executionState. For now we are assigning a round robin pattern per node.
    if (assignStatic) {
      for (auto it : currentAssignment) {
        auto &nodeAssignments = assignments.at(it.first);
        auto newAssignmentIdx = (it.second + 1) % nodeAssignments.size();
        auto newAssignment = nodeAssignments[newAssignmentIdx];
        assignment[it.first] = newAssignment;
        currentAssignment[it.first] = newAssignmentIdx;
      }
    }
    newState->init(deviceManagers_, assignment);
    pool->addNewState(std::move(newState));
  }
  states_[root] = std::move(pool);
}

void ThreadPoolExecutor::freePool(const DAGNode *root) { states_.erase(root); }

} // namespace runtime
} // namespace glow
