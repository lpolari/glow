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
#ifndef GLOW_RUNTIME_THREAD_POOL_EXECUTOR_H
#define GLOW_RUNTIME_THREAD_POOL_EXECUTOR_H

#include <condition_variable>
#include <mutex>
#include <unordered_map>

#include "NetworkExecutionState.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "glow/Runtime/Executor/Executor.h"

namespace glow {
namespace runtime {
using namespace std::chrono_literals;
using namespace std::chrono;
using clock_type = std::chrono::high_resolution_clock;

class ExecutionState;

/// This implementation of the Executor interface uses a thread pool to
/// handle and process multiple concurrent execution runs.
class ThreadPoolExecutor final : public Executor {
public:
  /// Constructor.
  explicit ThreadPoolExecutor(const DeviceManagerMapTy &deviceManagers,
                              unsigned numWorkers = kNumWorkers,
                              const std::string &name = "",
                              const std::vector<TimeslotBarrier*>
                                  *timeslotBarriers = NULL);

  /// Setup context pool for new network.
  void createPool(const DAGNode *root, unsigned poolSize,
                  bool assignStatic = false) override;

  /// Free the context pool for specified network.
  void freePool(const DAGNode *root) override;

  /// See Executor::run. A particular invocation is specified completely by
  /// the triple (roots, bindings, runId).
  void run(const DAGNode *root, std::unique_ptr<ExecutionContext> context,
           RunIdentifierTy runId, ResultCBTy cb) override;

  ~ThreadPoolExecutor() override { shutdown(); }

  void shutdown() override;

  const std::vector<TimeslotBarrier *> *getTimeslotBarriers() override{
    return this->timeslotBarriers_;
  }

  void setTimeslotSize(milliseconds timeslotSize) override {
    this->timeslotSize = timeslotSize;
  }

private:
  /// Execute the DAG node specified by \p node within the run corresponding to
  /// \p state.
  void executeDAGNode(NetworkExecutionState *executionState,
                      DAGNode *node,
                      unsigned criticality,
                      time_point<clock_type,milliseconds> deadline);

  void executeDAGNodeAsync(NetworkExecutionState *executionState,
                           DAGNode *node,
                           unsigned criticality,
                           time_point<clock_type, milliseconds> deadline);

  /// Handle the result returned asynchronously by the DeviceManager.
  /// \p executionState is tracks the state of the run that the node that
  /// finished executing belongs to, \p err is the Error returned by the
  /// DeviceManager, \p ctx is the ExecutionContext that contains the outputs
  /// produced by \p node during the run.
  ///
  /// The main purpose of this function is to help move computation off of the
  /// DeviceManager thread pool on onto the one owned by this class.
  void handleDeviceManagerResult(NetworkExecutionState *executionState,
                                 Error err,
                                 std::unique_ptr<ExecutionContext> ctx,
                                 DAGNode *node);

  /// The default number of workers in the thread pool.
  constexpr static unsigned kNumWorkers = 20;
  /// The thread pool used to drive execution.
  folly::CPUThreadPoolExecutor threadPool_;

  /// Map of networkExecutionState pools for each network.
  std::unordered_map<const DAGNode *,
                     std::unique_ptr<NetworkExecutionStatePool>>
      states_;

  const std::vector<TimeslotBarrier *> *timeslotBarriers_;

  /// Barrier for making sure all asynchronous requests made to the
  /// DeviceManager return before allowing destruction of the executor.
  InflightBarrier inflightBarrier_;
  /// Whether the executor is currently shutting down or not.
  std::atomic<bool> shuttingDown_{false};

  /// Map of available DeviceManagers.
  const DeviceManagerMapTy &deviceManagers_;

  /// Global Timeslot Size
  milliseconds timeslotSize = 0ms;
};

} // namespace runtime
} // namespace glow
#endif
