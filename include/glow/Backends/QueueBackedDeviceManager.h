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
#ifndef GLOW_BACKENDS_QUEUEBACKEDDEVICEMANAGER_H
#define GLOW_BACKENDS_QUEUEBACKEDDEVICEMANAGER_H

#include "glow/Backends/DeviceManager.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/ThreadPool.h"

#include <atomic>

namespace glow {
namespace runtime {
class QueueBackedDeviceManager : public DeviceManager {
protected:
  /// Thread which interfaces with the device.
  ThreadPool workThread_;

  /// Identifier for next run.
  std::atomic<RunIdentifierTy> nextIdentifier_{1};

public:
  QueueBackedDeviceManager(const DeviceConfig &config)
      : DeviceManager(config), workThread_(1) {}

  virtual ~QueueBackedDeviceManager() {
    ERR_TO_VOID(stop(true)); // will join workThread_
  }

  /// Initialize the device.
  Error init() override { return Error::success(); }

  /// Load the provided module into the device, readyCB will be called when
  /// ready to use
  void addNetwork(const Module *module, FunctionMapTy functions,
                  ReadyCBTy callback) override {
    workThread_.submit([this, module, f = std::move(functions),
                        c = std::move(callback)]() mutable {
      addNetworkImpl(module, std::move(f), std::move(c));
    });
  }

  /// Remove (and delete) the provided network and all it's functions, freeing
  /// up space on the device.
  void evictNetwork(std::string functionName,
                    EvictFunctionCBTy evictCB) override {
    workThread_.submit([this, functionName, evictCB] {
      evictNetworkImpl(functionName, evictCB);
    });
  }

  /// Execute the named Function in an already provided network on the device.
  /// functionName must match the name of a function already added.
  /// The ExecutionContext's PlaceholderBindings should have all Placeholders
  /// allocated. resultCB will be called with the ExecutionContext containing
  /// output tensors filled, and any generated TraceEvents.
  RunIdentifierTy runFunction(std::string functionName,
                              std::unique_ptr<ExecutionContext> context,
                              TimeslotBarrier* timeslotBarrier,
                              ResultCBTy callback) override {
    auto deadline = context->getNextDeadline();
    RunIdentifierTy id = nextIdentifier_++;
    size_t criticality = context->getCriticality();

    std::string* functionNameMem = new std::string(functionName.c_str());

    std::future<void> future = workThread_.submit(
        [this, id, functionName = std::move(functionName),
         context = std::move(context),
         callback = std::move(callback)]() mutable {
      runFunctionImpl(id, std::move(functionName), std::move(context),
                      std::move(callback));
    });

    // The function has been added to the execution queue. If we added
    // a critical function, it is now save for the less critical ones
    // decrement the timeslotBarriers for less critical functions
    if (criticality > 0 ){
      LOG(INFO) << "[" << *functionNameMem << "]"
                << " Decrement barrier for FRAME "
                << timeslotBarrier->getFramenumber() << "\n";
      timeslotBarrier->decrement();
    }

    // Stop the execution if it times out
    if (future.wait_until(deadline) != std::future_status::timeout)
    {
      // this will propagate exception from f() if any
      future.get();
    }
    else
    {
      this->continue_execution = 0;
      LOG(INFO) << "[" << *functionNameMem << "] Terminate the execution";
      future.wait();
      future.get();
      //throw std::runtime_error("Timeout")T;
    }
    if (this->reset_continue_execution_flag){
      this->continue_execution = 1;
      this->reset_continue_execution_flag = 0;
    }

    return id;
  }

  /// Stops execution and shuts down the Device.
  Error stop(bool block = true) override {
    workThread_.stop(block);
    return Error::success();
  }

protected:
  /// Operator handling methods to be implemented in subclasses (i.e. per Device
  /// type).

  /// Load and compile the Module.
  virtual void addNetworkImpl(const Module *, FunctionMapTy, ReadyCBTy) = 0;

  /// Remove the module and reclaim its memory.
  virtual void evictNetworkImpl(std::string functionName,
                                EvictFunctionCBTy evictCB) = 0;

  /// Execute provided Function.
  virtual void runFunctionImpl(RunIdentifierTy, std::string,
                               std::unique_ptr<ExecutionContext>,
                               ResultCBTy) = 0;
};
} // namespace runtime
} // namespace glow

#endif // GLOW_BACKENDS_QUEUEBACKEDDEVICEMANAGER_H
