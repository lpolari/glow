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
#ifndef GLOW_BACKENDS_COMPILEDFUNCTION_H
#define GLOW_BACKENDS_COMPILEDFUNCTION_H

#include "glow/Backend/BackendUtils.h"
#include "glow/ExecutionContext/ExecutionContext.h"
#include "glow/Graph/Nodes.h"
#include "glow/Support/Error.h"

namespace glow {

class PlaceholderBindings;
/// Interface for executing a compiled function.
class CompiledFunction {
public:
  /// Default Ctor.
  CompiledFunction() = delete;

  /// Ctor that accepts runtimeBundle.
  CompiledFunction(runtime::RuntimeBundle &&bundle);

  /// Dtor.
  virtual ~CompiledFunction();
  /// Execute the network and allocate Placeholder memory with given
  /// \p bindings providing mapping between Placeholder and populated tensor.
  /// \returns an Error if an error ocurred during execution.
  virtual Error execute(ExecutionContext *context) = 0;

  /// Getter for the runtimeBundle.
  runtime::RuntimeBundle &getRuntimeBundle() { return runtimeBundle_; }

  /// Collects constants for runtime.
  virtual void collectConstants(const Module *){};

  /// Setter for TraceEvent lookup. Note: does not enable tracing automatically.
  void setTraceInfo(TraceInfo &&info) { traceInfo_ = std::move(info); }

  /// Getter for the TraceEvent lookup.
  TraceInfo &getTraceInfo() { return traceInfo_; }
  const TraceInfo &getTraceInfo() const { return traceInfo_; }

  /// Read trace events out of this func and write them into /p bindings
  virtual void translateTraceEvents(ExecutionContext *bindings) const {}

  /// \returns the backend name used to compile this function.
  virtual std::string getCompileBackendName() const = 0;

  /// Once the compiledFunction is done being added to devices calling this
  /// method will free any resources needed to load the network on the device
  /// but not needed for running on the device.
  virtual void freeCompilationResources(){};

  ///  return wcet90 for that function
  uint64_t getWCET90(){
      return this->wcet90;
  };

  ///  return wcet95 for that function
  uint64_t getWCET95(){
    return this->wcet95;
  };

  ///  return wcet100 for that function
  uint64_t getWCET100(){
    return this->wcet100;
  }

  ///  return wcet105 for that function
  uint64_t getWCET105(){
    return this->wcet105;
  }

  ///  return wcet110 for that function
  uint64_t getWCET110(){
    return this->wcet110;
  }

  ///  return timeslot offset
  std::vector<uint64_t> getTimeslotOffsets(){
    return this->timeslot_offsets;
  }

  ///  set wcet90 for that function
  void setWCET90(uint64_t wcet90){
    this->wcet90 = wcet90;
  };

  ///  set wcet95 for that function
  void setWCET95(uint64_t wcet95){
    this->wcet95 = wcet95;
  };

  ///  set wcet100 for that function
  void setWCET100(uint64_t wcet100){
    this->wcet100 = wcet100;
  }

  void setWCET105(uint64_t wcet105){
    this->wcet105 = wcet105;
  }

  void setWCET110(uint64_t wcet110){
    this->wcet110 = wcet110;
  }

  ///  set timeslot offset
  void setTimeslotOffsets(std::vector<uint64_t> timeslot_offsets){
    this->timeslot_offsets = timeslot_offsets;
  }

protected:
  /// Contains symbol offsets and allocation sizes.
  runtime::RuntimeBundle runtimeBundle_;

  /// Information regarding runtime trace instrumentation present in this
  /// function.
  TraceInfo traceInfo_;

  uint64_t wcet90;
  uint64_t wcet95;
  uint64_t wcet100;
  uint64_t wcet105;
  uint64_t wcet110;

  std::vector<uint64_t> timeslot_offsets;
};
} // end namespace glow

#endif // GLOW_BACKENDS_COMPILEDFUNCTION_H
