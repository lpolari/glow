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
 * limitations under the License.gpu_0
 */

#include "benchmark/benchmark.h"

#include "glow/Backends/DeviceManager.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Runtime/Executor/ThreadPoolExecutor.h"
#include "glow/Runtime/HostManager/HostManager.h"
#include "nlohmann/json.hpp"
#include <fstream>

#include "CPUBackend.h"

#include <future>
#include <glow/Importer/Caffe2ModelLoader.h>

using json = nlohmann::json;
using namespace std::chrono_literals;

#ifndef GLOW_DATA_PATH
#define GLOW_DATA_PATH
#endif

using namespace glow;
using namespace glow::runtime;


//===--------------------------------------------------------------------===//
//                           Input Parameters                               //
//===--------------------------------------------------------------------===//

static size_t period = atoi(getenv("NET_PERIOD"));
static std::string netname = getenv("NET_NAME");
static const char* inputName = getenv("NET_INPUT_NAME");

//===--------------------------------------------------------------------===//
//                       Common Utility Functions                           //
//===--------------------------------------------------------------------===//

static int current_run = 0;

double customStatisticHandler(const std::vector<double>& v){
  std::string partition_config_filename =
      std::string("../data/adas-pb-output/partition_configs/partition_config_"
                  + netname + "_" + std::to_string(period)) + ".json";
  std::string runtime_dump_filename =
      std::string("../data/adas-pb-output/runtime_dumps/"
                  + netname + "_" + std::to_string(period)) + ".txt";

  LOG(INFO) << partition_config_filename;
  std::ifstream ifs(partition_config_filename);
  json jf = json::parse(ifs);

  std::ofstream runtime_dump_file;
  runtime_dump_file.open(runtime_dump_filename, std::ios_base::app);
  runtime_dump_file << "-------------- Partition "
                    << current_run+1
                    << " -------------\n";

  for (auto i = v.begin(); i != v.end(); ++i){
    size_t rt = (size_t) (*i * 1000000);
    std::string s = std::to_string(rt) + ", ";
    runtime_dump_file << s;
  }
  runtime_dump_file.close();

  std::vector<double> v_copy(v);
  std::sort(v_copy.begin(), v_copy.end());
  int index_90 = 9* (v_copy.size()/10) -1;
  int index_95 = 95* (v_copy.size()/100) -1;

  double res_90 = v_copy[index_90];
  double res_95 = v_copy[index_95];
  double res_100 = v_copy[v_copy.size()-1];
  double res_105 = 1.05 * res_100;
  double res_110 = 1.1 * res_100;

  jf["wcet90"][(current_run)/2] = res_90*1000;
  jf["wcet95"][(current_run)/2] = res_95*1000;
  jf["wcet100"][(current_run)/2] = res_100*1000;
  jf["wcet105"][(current_run)/2] = res_105*1000;
  jf["wcet110"][(current_run)/2] = res_110*1000;
  current_run++;
  std::ofstream file(partition_config_filename);
  file << jf;

  return res_100;
}

//===--------------------------------------------------------------------===//
//              Benchmark Declaration and Instantiation Macros              //
//===--------------------------------------------------------------------===//


/// Declare a subclass of an arbitrary Benchmark class and override its
/// setUpModule method with the given moduleCreator function.
#define DECLARE_RUNTIME_COMPONENT_BENCHMARK(name, moduleCreator, component)    \
  template <typename BackendTy>                                                \
  class name##component##Benchmark : public component##Benchmark<BackendTy> {  \
  protected:                                                                   \
    void setUpModule(benchmark::State &state) override {                       \
      this->mod_ = moduleCreator();                                            \
    }                                                                          \
  };

/// Define a RuntimeBenchmark subclass declared using
/// DECLARE_XXX_BENCHMARK for a specific backend and component. This instance
/// calls RuntimeBenchmark::runBenchmark to run the benchmark.
#define INSTANTIATE_RUNTIME_COMPONENT_BENCHMARK(name, backend, component, func_num) \
  BENCHMARK_TEMPLATE_DEFINE_F(name##component##Benchmark, component##backend,  \
                              backend)                                         \
  (benchmark::State & state) { runBenchmark(state); }                          \
  BENCHMARK_REGISTER_F(name##component##Benchmark, component##backend)         \
      ->Unit(benchmark::kMillisecond)                                          \
      ->MeasureProcessCPUTime()                                                \
      ->DenseRange(1, period)                                                  \
      ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {  \
        return customStatisticHandler(v);                                      \
      });

//===--------------------------------------------------------------------===//
//                  Benchmark Template Fixture Classes                      //
//===--------------------------------------------------------------------===//

// Benchmark for one subnet, specified in the state parameter
// on the DeviceManager-Level
template <typename BackendTy>
class SubnetBenchmark : public benchmark::Fixture {
public:
  void SetUp(benchmark::State &state) {
    setUpBackend(state);
    setUpModule(state);
    setUpExecutionContext(state);
    setUpHostManager(state);
  }

protected:

  std::unique_ptr<Backend> &getBackend() { return backend_; }
  void setUpBackend(benchmark::State &state) {
    backend_ = glow::make_unique<BackendTy>();
  }

  std::unique_ptr<Module> &getModule() { return mod_; }

  virtual void setUpModule(benchmark::State &state) = 0;

  std::unique_ptr<ExecutionContext> &getExecutionContext() { return ctx_; }
  virtual void setUpExecutionContext(benchmark::State &state) {
    // Allocate all Placeholders in mod_ and move the bindings into an
    // ExecutionContext object.
    auto bindings = glow::make_unique<PlaceholderBindings>();

    // LPolariToDo Hacky way to check if module was already set since we
    // only check if mod_ is a valid parameter which is only true for
    // the first run. Maybe create a setup boolean variable
    if (mod_) {
      placeholderList = mod_->getPlaceholders();
      bindings->allocate(placeholderList);
      ctx_ = glow::make_unique<ExecutionContext>(std::move(bindings));
      ctx_->setTraceContext(
          glow::make_unique<TraceContext>(TraceLevel::STANDARD));
    }
  }

  /// Setup the HostManager with the given module
  /// Get the deviceManager from the HostManager
  virtual void setUpHostManager(benchmark::State &state) {

    // Only execute HostManager/DeviceManager setup for the first instance
    // of the SubnetBenchmark, since all subnetBenchmarks shall use the same
    // HostManager/DeviceManager. Therefor these are created as static
    // members
    if (setup++){
      return;
    }

    setup = true;
    std::unique_ptr<Backend> &backend = this->getBackend();
    std::unique_ptr<Module> &mod = this->getModule();

    // Check that the backend is valid.
    if (!backend) {
      state.SkipWithError(
          "Unable to set up host manager - backend not set up!");
      return;
    }

    // Check that the module is valid.
    if (!mod) {
      state.SkipWithError("Unable to set up host manager - module not set up!");
      return;
    }

    // Create DeviceConfigs with which to initialize the HostManager
    // instance.
    std::vector<std::unique_ptr<DeviceConfig>> configs;
    for (unsigned i = 0; i < numDeviceManagers_; ++i) {
      configs.emplace_back(
          glow::make_unique<DeviceConfig>(backend->getBackendName()));
    }

    // Create and initialize the HostManager instance.
    hostManager_ = glow::make_unique<HostManager>(std::move(configs));

    // Remember the names of all functions in the module before passing
    // ownership to the HostManager.
    for (auto *function : mod->getFunctions()) {
      functions_.emplace_back(function->getName());
    }

    // Add the module to the HostManager instance.
    CompilationContext cctx;
    cctx.period=period;
    //cctx.backendOpts.autoInstrument = true;
    bool error = ERR_TO_BOOL(hostManager_->addNetwork(std::move(mod), cctx));
    if (error) {
      state.SkipWithError("Unable to set up host manager - failed to add "
                          "module!");
    }

    this->deviceManager_ = hostManager_->getDeviceManager();
  }


  void runBenchmark(benchmark::State &state) {
    std::unique_ptr<ExecutionContext> &ctx = this->getExecutionContext();

    for (auto _ : state) {
      std::promise<void> promise;
      std::future<void> future = promise.get_future();
      LOG(INFO) << "STATE RANGE 0 " << state.range(0) << "\n";
      std::string name = netname + "_id<id>_part";
      name.append(std::to_string(state.range(0)));
      deviceManager_->runFunction(
          name, std::move(ctx),
          [&promise, &ctx](runtime::RunIdentifierTy /*runId*/, Error err,
                           std::unique_ptr<ExecutionContext> result) {
          // We don't care about the result but check the error to avoid
          // uncheck error error.
          ERR_TO_BOOL(std::move(err));
          ctx = std::move(result);
          promise.set_value();
        });
      future.wait();
    }
  }


  /// An instance of the Backend the benchmark is running against.
  std::unique_ptr<Backend> backend_;
  /// The module to use for the benchmark.
  std::unique_ptr<Module> mod_;
  /// The execution context to use for the benchmark.
  static std::unique_ptr<ExecutionContext> ctx_;
  /// List of placeholder tensors
  PlaceholderList placeholderList;
  /// The HostManager instance to handle compilation and partitioning
  static std::unique_ptr<HostManager> hostManager_;
  /// The number of DeviceManagers to use during the benchmark.
  static constexpr unsigned numDeviceManagers_{1};
  /// List of functions in the module.
  std::vector<std::string> functions_;
  /// The DeviceManager instance being benchmarked.
  static std::shared_ptr<DeviceManager> deviceManager_;
  /// true if static members as HostManager or DeviceManager were already setup
  static size_t setup;

};

template <typename BackendTy>
std::unique_ptr<ExecutionContext> SubnetBenchmark<BackendTy>::ctx_ = NULL;
template<typename BackendTy>
std::unique_ptr<HostManager> SubnetBenchmark<BackendTy>::hostManager_ = NULL;
template<typename BackendTy>
std::shared_ptr<DeviceManager> SubnetBenchmark<BackendTy>::deviceManager_ = NULL;
template<typename BackendTy>
size_t SubnetBenchmark<BackendTy>::setup = 0;


//===--------------------------------------------------------------------===//
//              Benchmark Module and DAG Creator Functions                  //
//===--------------------------------------------------------------------===//

static size_t setup = 0;

// Create the base module via Loader
std::unique_ptr<Module> CreateBaseModule() {
  if (setup){return nullptr;}
  setup = 1;

  std::vector<dim_t> inputShape{1, 3, 224, 224};
  std::unique_ptr<Module> module = glow::make_unique<Module>();
  TypeRef inputType = module->uniqueType(ElemKind::FloatTy, inputShape);
  Function *F = module->createFunction(netname + "_id<id>");

  // Load a model
  std::string source_prefix = "../data/onnx-models/" + netname + "/";
  std::string NetDescFilename(GLOW_DATA_PATH source_prefix + "predict_net.pb");
  std::string NetWeightFilename(GLOW_DATA_PATH source_prefix + "init_net.pb");

  Placeholder *output;
  Caffe2ModelLoader caffe2LD(NetDescFilename, NetWeightFilename, {inputName},
                             {inputType}, *F);
  output = EXIT_ON_ERR(caffe2LD.getSingleOutput());
  return std::move(module);
}

//--------------------------------------------------------------------------//

//===--------------------------------------------------------------------===//
//              Benchmark Declarations and Instantiations                   //
//===--------------------------------------------------------------------===//

DECLARE_RUNTIME_COMPONENT_BENCHMARK(ADAS, CreateBaseModule, Subnet)


// Instantiate the SingleNode benchmark for the CPU backend. This creates
// instances of the SubnetBenchmark, ExecutorBenchmark and
// DeviceManagerBenchmark subclasses declared by the macro above for the CPU
// backend.
INSTANTIATE_RUNTIME_COMPONENT_BENCHMARK(ADAS, CPUBackend,
                                        Subnet, 2)

//===--------------------------------------------------------------------===//
//                           Benchmark Main                                 //
//===--------------------------------------------------------------------===//

// Benchmark main.
int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
}

int main(int, char**);
