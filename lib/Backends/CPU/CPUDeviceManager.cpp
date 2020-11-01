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
#include "CPUDeviceManager.h"
#include "CPUFunction.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace glow {
namespace runtime {

unsigned GlowCPUMemory = 0;

static llvm::cl::opt<unsigned, /* ExternalStorage */ true> GlowCPUMemoryOpt(
    "cpu-memory",
    llvm::cl::desc("CPU DeviceManager maximum memory in kilobytes."),
    llvm::cl::location(GlowCPUMemory));

DeviceManager *createCPUDeviceManager(const DeviceConfig &config) {
  if (GlowCPUMemory) {
    // Convert command line GlowCPUMemory to bytes from kilobytes.
    auto configNew = config;
    configNew.setDeviceMemory(uint64_t{GlowCPUMemory} * 1024);
    return new CPUDeviceManager(configNew);
  }
  return new CPUDeviceManager(config);
}

uint64_t CPUDeviceManager::getMaximumMemory() const { return maxMemoryBytes_; }

uint64_t CPUDeviceManager::getAvailableMemory() const {
  return maxMemoryBytes_ - usedMemoryBytes_;
}

std::vector<uint64_t>
    CPUDeviceManager::getAvailableLoadPerTimeslot(uint64_t p) const {
      return availableLoadPerTimeslot.at(p);
}

bool CPUDeviceManager::isMemoryAvailable(uint64_t estimate) const {
  // No fuzz factor for the CPU device.
  return maxMemoryBytes_ >= (usedMemoryBytes_ + estimate);
}

DeviceInfo CPUDeviceManager::getDeviceInfo() const {
  // TODO: these may need to be tweaked depending on specific CPU.
  DeviceInfo info = DeviceInfo();
  info.sramCapacity = 256 * 1024 * 1024;
  info.peakCompute = 2.2 * 1024 * 1024 * 1024 * 1024;
  info.peakDramBw = 110.0 * 1024 * 1024 * 1024;
  info.peakSramBw = 1024.0 * 1024 * 1024 * 1024;
  info.peakPCIeBw = 16.0 * 1024 * 1024 * 1024;
  return info;
}

void CPUDeviceManager::addNetworkImpl(const Module *module,
                                      FunctionMapTy functions,
                                      ReadyCBTy readyCB) {
  DCHECK(readyCB != nullptr);

  uint64_t allFunctionsMemoryBytes{0};

  std::string continue_execution_flag_name = "";

  // First check for uniqueness of the function name.
  for (const auto &func : functions) {

    if (continue_execution_flag_name == ""){
      // Set the flag name to stop the inference
      // the flag name is derived from the function prefix which
      // corresponds to the module name
      std::string function_name = func.first;
      int found = function_name.find_last_of("_");
      std::string module_name = function_name.substr(0,found);
      std::string continue_execution_flag_name = "global_" + module_name
                                                 + "_continue";
      llvm::sys::DynamicLibrary::AddSymbol(continue_execution_flag_name,
                                           &this->continue_execution);
    }

    if (functions_.count(func.first) != 0) {
      readyCB(
          module,
          MAKE_ERR(
              llvm::formatv(
                  "Failed to add network: already have a function called {0}",
                  func.first)
                  .str()));
      return;
    }

    if (func.second->getCompileBackendName() != "CPU") {
      readyCB(
          module,
          MAKE_ERR(
              llvm::formatv(
                  "Failed to add network: function {0} is not a CPUFunction",
                  func.first)
                  .str()));
      return;
    }


    for (uint64_t offset : func.second->getTimeslotOffsets()){
      this->availableLoadPerTimeslot.at(90)[offset] -= func.second->getWCET90();
      this->availableLoadPerTimeslot.at(95)[offset] -= func.second->getWCET95();
      this->availableLoadPerTimeslot.at(100)[offset] -= func.second->getWCET100();
      this->availableLoadPerTimeslot.at(105)[offset] -= func.second->getWCET105();
      this->availableLoadPerTimeslot.at(110)[offset] -= func.second->getWCET110();
    }

    allFunctionsMemoryBytes +=
        func.second->getRuntimeBundle().getConstantWeightSize();
  }

  if (usedMemoryBytes_ + allFunctionsMemoryBytes > maxMemoryBytes_) {
    readyCB(module,
            MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_OUT_OF_DEVICE_MEMORY,
                     "Failed to add network: not enough memory"));
    return;
  }

  // Add to the function name lookup map.
  for (const auto &func : functions) {
    if (func.second->getRuntimeBundle().getConstants() == nullptr) {
      func.second->getRuntimeBundle().collectConstants(module);
    }
    functions_.emplace(func.first, func.second);
  }

  usedMemoryBytes_ += allFunctionsMemoryBytes;
  assert(usedMemoryBytes_ <= maxMemoryBytes_);

  // Export change in memory usage.
  exportMemoryCounters();

  // Fire the ready CB.
  readyCB(module, Error::success());
}

void CPUDeviceManager::evictNetworkImpl(std::string functionName,
                                        EvictFunctionCBTy evictCB) {
  DCHECK(evictCB != nullptr);

  auto it = functions_.find(functionName);
  if (it != functions_.end()) {
    usedMemoryBytes_ -= it->second->getRuntimeBundle().getConstantWeightSize();
    functions_.erase(it);
  } else {
    evictCB(functionName,
            MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                     strFormat("Could not find function with name %s to evict",
                               functionName.c_str())));
    return;
  }
  // Export change in memory usage.
  exportMemoryCounters();

  evictCB(functionName, Error::success());
}

void CPUDeviceManager::runFunctionImpl(
    RunIdentifierTy id, std::string function,
    std::unique_ptr<ExecutionContext> context, ResultCBTy resultCB) {
  DCHECK(resultCB != nullptr);


  std::size_t found = function.find("_");
  std::string net_name = function.substr(0, found);
  std::string name = "DeviceManager::run(" + function + ")";
  TRACE_EVENT_SCOPE_NAMED(context->getTraceContext(), TraceLevel::RUNTIME,
                          name, dmRun);
  if (context->getTraceContext()) {
    context->getTraceContext()->setThreadName("CPU DeviceManager");
  }

  if (this->continue_execution == 0) {
    int queueSize = this->workThread_.getQueueEmpty();
    if (!queueSize){
      LOG(INFO) << "%% Queue is Empty -> Set to one again " << queueSize;
      this->continue_execution = 1;
    } else {
      LOG(INFO) << "%% Queue is Not EMTPY -> reset continue flag later "
                << queueSize;
    }

    dmRun.addArg("reason", "Deadline already reached before start");
    TRACE_EVENT_SCOPE_END_NAMED(dmRun);
    resultCB(id,
             MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_EXECUTION_TIMEOUT,
                      llvm::formatv(
                          "Deadline for {0} already reached before start",
                          function).str()),
             std::move(context));
    return;
  }

  auto funcIt = functions_.find(function);
  if (funcIt == functions_.end()) {
    dmRun.addArg("reason", "function not found");
    TRACE_EVENT_SCOPE_END_NAMED(dmRun);
    resultCB(id,
             MAKE_ERR(ErrorValue::ErrorCode::RUNTIME_NET_NOT_FOUND,
                      llvm::formatv("Function {0} not found", function).str()),
             std::move(context));
    return;
  }

  CompiledFunction *func = funcIt->second;

  // Run that function.
  auto executeErr = func->execute(context.get());

  // If the timeout for the frame has been reached, we have to reset continue
  // execution for the next frame; But not if there are still subnets
  // from this frame left. These will stop immediately and reset the value
  // later
  int queueSize = this->workThread_.getQueueEmpty();
  if (!queueSize){
    LOG(INFO) << "%% Queue is Empty -> Set to one again " << queueSize;
    this->reset_continue_execution_flag = 1;
  } else {
    LOG(INFO) << "%% Queue is Not EMTPY -> Reset continue flag later "
              << queueSize;
  }

  if(executeErr){
    dmRun.addArg("reason", "Deadline reached during execution");
  }

  // End the TraceEvent early to avoid time in the CB.
  TRACE_EVENT_SCOPE_END_NAMED(dmRun);

  // Fire the resultCB.
  resultCB(id, std::move(executeErr), std::move(context));
}
} // namespace runtime
} // namespace glow
