//
// Created by parmla1 on 27.03.20.
//

#ifndef GLOW_LPOLARI_RUNTIME_H
#define GLOW_LPOLARI_RUNTIME_H

#include "glow/Base/Image.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Runtime/HostManager/HostManager.h"
#include "glow/Runtime/RuntimeTypes.h"
#include "glow/Support/Error.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include <thread>
#include <iostream>


#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

#include <glog/logging.h>

#include <chrono>
#include <future>

using namespace glow;
using namespace glow::runtime;

typedef struct NetworkDescription {
  size_t id;
  size_t period;
  size_t criticality;
  std::string name;
  std::string netDescFilename;
  std::string netWeightFilename;
  std::string partitionConfigFilename;
  std::string inputName;
  std::unique_ptr<CompilationContext> ctx;
  std::unique_ptr<Module> module;
  Placeholder *input;
  std::unique_ptr<ExecutionContext> ectx;
  std::string current_input_path;
  PlaceholderList phList;
  std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>
      current_release_time;
  std::vector<size_t> runtime_frames;
  std::vector<size_t> runtime_ms;
  std::promise<void> finished;
  std::atomic<int> returned;
  std::atomic<int> ready;
} NetDesc, *pNetDesc;


#endif // GLOW_LPOLARI_RUNTIME_H
