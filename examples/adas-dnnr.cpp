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
 * See the License for the specific slanguage governing permissions and
 * limitations under the License.
 */

//
// Created by lpolari on 27.03.2020.
//


#include "adas-dnnr.h"
#include "llvm/Support/DynamicLibrary.h"
#include <iostream>
#include <fstream>

using namespace std::literals;
using namespace std::chrono;
using clock_type = std::chrono::high_resolution_clock;


namespace {
llvm::cl::OptionCategory category("adas-dnnr options");

llvm::cl::opt<std::string>
    scheduling_policy("scheduling",
                      llvm::cl::desc("Scheduling algorithm for dnn execution"),
                      llvm::cl::init("MCDTS"),
                      llvm::cl::value_desc("policy"),
                      llvm::cl::cat(category));

llvm::cl::opt<unsigned>
    p_LO("p-lo",
                      llvm::cl::desc("p value for wcet quantile (low criticalities)"),
                      llvm::cl::init(95),
                      llvm::cl::value_desc("p"),
                      llvm::cl::cat(category));

llvm::cl::opt<unsigned>
    p_HI("p-hi",
                      llvm::cl::desc("p value for wcet quantile (high criticalities)"),
                      llvm::cl::init(105),
                      llvm::cl::value_desc("p"),
                      llvm::cl::cat(category));

#define TIMESLOT_NOT_SPECIFIED 99999
llvm::cl::opt<unsigned>
    timeslot_size("timeslot-size",
                      llvm::cl::desc("Time between successive frames"),
                      llvm::cl::init(TIMESLOT_NOT_SPECIFIED),
                      llvm::cl::value_desc("ms"),
                      llvm::cl::cat(category));

llvm::cl::opt<std::string>
    inputDirectory(llvm::cl::desc("input directory for images, which must be "
                                  "png's with standard imagenet normalization"),
                   llvm::cl::init("../data/images/imagenet"),
                   llvm::cl::Positional, llvm::cl::cat(category));

llvm::cl::opt<unsigned> numDevices("num-devices",
                                   llvm::cl::desc("Number of Devices to use"),
                                   llvm::cl::init(2), llvm::cl::value_desc("N"),
                                   llvm::cl::cat(category));
llvm::cl::opt<unsigned>
    maxFrames("max-images",
              llvm::cl::desc("Maximum number of images to load and classify"),
              llvm::cl::init(8), llvm::cl::value_desc("N"),
              llvm::cl::cat(category));

llvm::cl::opt<std::string> tracePath("trace-path",
                                     llvm::cl::desc("Write trace logs to disk"),
                                     llvm::cl::init(""),
                                     llvm::cl::cat(category));
llvm::cl::opt<std::string>
    backend("backend",
            llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
            llvm::cl::Optional, llvm::cl::init("CPU"), llvm::cl::cat(category));

llvm::cl::opt<bool>
    autoInstrument("auto-instrument",
                   llvm::cl::desc("Add instrumentation for operator tracing"),
                   llvm::cl::Optional, llvm::cl::init(false),
                   llvm::cl::cat(category));

llvm::cl::opt<unsigned> random_seed("random-seed",
                                   llvm::cl::desc("random seed to generate network parameters"),
                                   llvm::cl::init(0), llvm::cl::value_desc("N"),
                                   llvm::cl::cat(category));

std::mutex eventLock;
std::unique_ptr<TraceContext> traceContext;
std::mutex frameIncrementLock;
Tensor batch;
std::string path;
const size_t TIMESLOT_BARRIER_Z = 7;

} // namespace

bool netDescSorter(NetDesc* lhs, NetDesc* rhs) {
  return lhs->criticality < rhs->criticality;
}

size_t started = 0;
time_point<clock_type , milliseconds> target_time;
milliseconds timeslotSize;


/// Loads the model into /p module and returns the input and output
/// Placeholders. Appending count to the function name.
Placeholder *loadModel(const std::string &name,
                       const std::string &netDescFilename,
                       const std::string &netWeightFilename,
                       TypeRef inputType,
                       std::string inputName,
                       Module *module, unsigned int id) {
  Function *F = module->createFunction(name + "_id" + std::to_string(id));

  LOG(INFO) << "Loading " << name << " model.";

  const char* c_inputName = inputName.c_str();
  Caffe2ModelLoader loader(netDescFilename, netWeightFilename,
                           {c_inputName}, {inputType}, *F);
  Placeholder *input = llvm::cast<Placeholder>(
      EXIT_ON_ERR(loader.getNodeValueByName(inputName)));
  return input;
}

/// Starts a run of the specified dnn on the given image.
/// The image must be already loaded into the input placeholder
/// the next instance of the dnn in started automatically
/// in the callback
void dispatchRun(NetworkDescription* netDesc, HostManager *hostManager) {
  LOG(INFO) << "Dispatch " << netDesc->name
            << " Period " << netDesc->period
            << " Offset " << netDesc->ectx->getOffset()
            << " Target time " << netDesc->ectx->getNextDeadline()
                .time_since_epoch().count();

  auto runid = hostManager->runNetwork(
      netDesc->name + "_id" + std::to_string(netDesc->id),
      std::move(netDesc->ectx),
      [hostManager, netDesc](RunIdentifierTy runid, Error err,
                                   std::unique_ptr<ExecutionContext> context) {
        LOG(INFO) << "[" + netDesc->name + "_id" + std::to_string(netDesc->id)
                         + "] Enter network callback" << "\n";

        //auto *bindings = context->getPlaceholderBindings();

        // Read output tensor
        /*
        size_t maxIdx =
            bindings->get(bindings->getPlaceholderByName("gpu_0_softmax"))
                ->getHandle()
                .minMaxArg()
                .second;

        LOG(INFO) << "[" + netDesc->name + "_id" + std::to_string(netDesc->id)
                     + "] OUTPUT :: (" << runid << ") "
                  << netDesc->current_input_path << ": " << maxIdx << "\n";
        */

        if (!tracePath.empty()) {
          std::lock_guard<std::mutex> l(eventLock);
          // Merge this run's TraceEvents into the global TraceContext.
          traceContext->merge(context->getTraceContext());
        }


        frameIncrementLock.lock();
        auto current_release_time = context->getNextDeadline() - timeslotSize;
        netDesc->ectx = std::move(context);
        updateInputPlaceholders(*(netDesc->ectx->getPlaceholderBindings()),
                                {netDesc->input}, {&batch});
        netDesc->current_input_path = path;
        auto last_release_time = netDesc->current_release_time;
        netDesc->current_release_time = current_release_time;
        auto last_runtime =
            netDesc->current_release_time - last_release_time;
        auto last_runtime_in_frames =
            last_runtime.count() / timeslotSize.count();
        netDesc->runtime_ms.push_back(last_runtime.count());
        netDesc->runtime_frames.push_back(last_runtime_in_frames);
        netDesc->ectx->setTraceContext(
            glow::make_unique<TraceContext>(TraceLevel::STANDARD));
        dispatchRun(netDesc, hostManager);
        netDesc->ready = 1;
        frameIncrementLock.unlock();
      });
  LOG(INFO) << "Started run ID: " << runid;
}


std::vector<NetworkDescription*> setupNetworks(){
  std::vector<NetworkDescription*> nets;

  std::vector<std::string> net_names{
      "resnet50",
      "shufflenet",
      "inception_v1",
      "inception_v2",
      "densenet121",
      "vgg19"
  };

  srand(random_seed);
  for (int i=0; i<4; i++){
    NetworkDescription* net = new NetworkDescription();
    size_t period = 1 + (rand() % 3);
    if (period == 3){
      period++;
    }
    size_t net_index = rand() % 4;
    net->name = net_names.at(net_index);
    net->period = period;
    net->netDescFilename = "../data/onnx-models/" + net_names[net_index] + "/predict_net.pb";
    net->netWeightFilename = "../data/onnx-models/" + net_names[net_index] + "/init_net.pb";
    net->partitionConfigFilename = "../data/adas-pb-output/partition_configs/partition_config_"
                                   + net_names[net_index] + "_"
                                   + std::to_string(period) + ".json";

    // networks above index 2 use a different input name
    if (net_index < 2){
      net->inputName = "gpu_0/data";
    } else{
      net->inputName = "data";
    }

    net->id = i;
    if (i<2){
      net->criticality = 2;
    } else {
      net->criticality = 0;
    }
    nets.push_back(net);
  }
  return nets;
}


int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "ADAS DNN Runtime Environment");

  timeslotSize = milliseconds(timeslot_size);
  std::vector<NetworkDescription*> nets = setupNetworks();
  std::sort(nets.rbegin(), nets.rend(), &netDescSorter);

  for (auto &&netDesc : nets){
    LOG(INFO) << "NET :: " << netDesc->name << " ID :: " << netDesc->id << "\n";
    std::atomic_init(&netDesc->returned, 0);
    std::atomic_init(&netDesc->ready, 1);
  }

  LOG(INFO) << "Initializing " << numDevices << " " << backend
            << " devices on HostManager.";

  std::vector<std::unique_ptr<DeviceConfig>> configs;
  for (unsigned int i = 0; i < numDevices; ++i) {
    auto config = glow::make_unique<DeviceConfig>(backend);
    config->setDeviceMemory(4000000000);
    config->setTimeslotSize(timeslot_size);
    configs.push_back(std::move(config));
  }

  std::unique_ptr<HostManager> hostManager =
      glow::make_unique<HostManager>(std::move(configs));

  // If tracing is enabled, create a TraceContext to merge each runs events
  // into.
  if (!tracePath.empty()) {
    traceContext = glow::make_unique<TraceContext>(TraceLevel::STANDARD);
  }

  // Load model, create a context, and add to HostManager.
  LOG(INFO) << "Loading files from " << inputDirectory;
  std::error_code code;
  llvm::sys::fs::directory_iterator dirIt(inputDirectory, code);
  if (code.value()) {
    LOG(ERROR) << "Couldn't read from directory: " << inputDirectory
               << " - code" << code.value() << "\n";
    exit(code.value());
  }

  std::vector<dim_t> inputShape{1, 3, 224, 224};

  path = dirIt->path();
  auto image = readPngImageAndPreprocess(
      path, ImageNormalizationMode::k0to1, ImageChannelOrder::BGR,
      ImageLayout::NCHW, imagenetNormMean, imagenetNormStd);


  for (auto &&netDesc : nets){
    Placeholder *input;
    PlaceholderList phList;

    netDesc->module = glow::make_unique<Module>();
    TypeRef inputType = netDesc->module->uniqueType(ElemKind::FloatTy,
                                                    inputShape);
    netDesc->input = loadModel(netDesc->name, netDesc->netDescFilename,
                               netDesc->netWeightFilename, inputType,
                               netDesc->inputName, netDesc->module.get(),
                               netDesc->id);
    netDesc->phList = netDesc->module->getPlaceholders();
    CompilationContext cctx;
    cctx.scheduling_policy = scheduling_policy;
    cctx.backendOpts.autoInstrument = autoInstrument;
    cctx.period = netDesc->period;
    cctx.criticality = netDesc->criticality;
    cctx.network_name = netDesc->name;
    cctx.network_id = netDesc->id;
    cctx.partitionConfigFilename = netDesc->partitionConfigFilename;
    cctx.p_LO = p_LO;
    cctx.p_HI = p_HI;
    cctx.scheduling_policy = scheduling_policy;

    EXIT_ON_ERR(hostManager->addNetwork(std::move(netDesc->module), cctx,
        /*saturateHost*/ true));

    std::unique_ptr<ExecutionContext> ectx =
        glow::make_unique<ExecutionContext>();

    ectx->setName(netDesc->name + "_id" + std::to_string(netDesc->id));
    ectx->setCriticality(netDesc->criticality);
    ectx->setPeriod(netDesc->period);
    ectx->setTraceContext(
        glow::make_unique<TraceContext>(TraceLevel::STANDARD));

    ectx->getPlaceholderBindings()->allocate(netDesc->phList);
    batch = image.getUnowned(inputShape);
    updateInputPlaceholders(*(ectx->getPlaceholderBindings()),
                            {netDesc->input}, {&batch});
    netDesc->current_input_path = path;
    netDesc->ectx = std::move(ectx);
  }

  /// if no timeslot size was specified set it to the minimum
  /// for the specified scheduling policy
  size_t overall_period_load = 0;
  if (timeslot_size == TIMESLOT_NOT_SPECIFIED) {
    timeslotSize = 0ms;

    /// calculate maximum high criticality load for MCDTS and RDTS algorithm
    /// for the MCDTS algorithm high criticality wcet estimates of non-critical
    /// DNNs are set to 0. Therefor <getDeviceLoadLeftPerTimeslot> only returns
    /// the load of the critical DNNs
    for (auto load_left : hostManager->getDeviceLoadLeftPerTimeslot(0, p_HI)) {
      /// calculate overall load as the minimal period for real-time
      /// scheduling algorithms
      overall_period_load += milliseconds(timeslot_size - load_left).count();
      if (timeslotSize < milliseconds(timeslot_size - load_left)) {
        timeslotSize = milliseconds(timeslot_size - load_left);
      }
    }
    LOG(INFO) << "Critical load in ms :: " << timeslotSize.count();

    /// calculate maximum low criticality load for the MCDTS algorithm
    if (scheduling_policy == "MCDTS") {
      for (auto load_left :
           hostManager->getDeviceLoadLeftPerTimeslot(0, p_LO)) {
        if (timeslotSize < milliseconds(timeslot_size - load_left)) {
          timeslotSize = milliseconds(timeslot_size - load_left);
        }
      }
    }
    LOG(INFO) << "Non-critical load in ms :: " << timeslotSize.count();
  }

  hostManager->setTimeslotSize(timeslotSize);
  LOG(INFO) << "Timeslot size in ms :: " << timeslotSize.count();
  maxFrames = 128;
  const std::vector<TimeslotBarrier*> *timeslotBarriers = hostManager->getTimeslotBarriers();

  auto when_started = clock_type::now();
  target_time = time_point_cast<milliseconds>(when_started + timeslotSize);

  while (started < maxFrames) {

    // Dispatch the first run of all added networks. Afterwards the callbacks
    // of each run will dispatch the next instance
    if (started == 0){
      for (auto &&netDesc : nets) {
        if (netDesc->ready) {
          netDesc->ready = 0;
          LOG(INFO) << "INITIAL TARGET TIME "
                    << target_time.time_since_epoch().count();
          netDesc->current_release_time
              = time_point_cast<milliseconds>(when_started);
          netDesc->ectx->setNextDeadline(target_time);
          netDesc->ectx->setOffset(started % 4);
          netDesc->ectx->setTraceContext(
              glow::make_unique<TraceContext>(TraceLevel::STANDARD));

          dispatchRun(netDesc, hostManager.get());
        }
      }
    }

    LOG(INFO) << "Signal Start of The Frame // "
              << "Decrement TimeslotBarrier at offset :: "
              << started << "\n";
    timeslotBarriers->at(started % 4)->decrement();

    // Load next image
    dirIt.increment(code);
    path = dirIt->path();
    image = readPngImageAndPreprocess(
        path, ImageNormalizationMode::k0to1,
        ImageChannelOrder::BGR,
        ImageLayout::NCHW, imagenetNormMean, imagenetNormStd);
    batch = image.getUnowned(inputShape);

    std::this_thread::sleep_until(target_time);
    frameIncrementLock.lock();
    LOG(INFO) << "Reset TimeslotBarrier at offset :: " << started;
    timeslotBarriers->at(started % 4)->increment(TIMESLOT_BARRIER_Z);
    target_time += timeslotSize;
    started++;
    frameIncrementLock.unlock();
  }

  double cpu_utilization;
  if (!tracePath.empty()) {
    traceContext->dump(tracePath, "adas-dnnr");
    size_t cpu_time = traceContext->getCPUUtilization() / 1000;
    size_t wall_time = 128 * timeslotSize.count();
    cpu_utilization = (double)cpu_time / (double)wall_time;
    LOG(INFO) << "CPU UTILIZATION :: " << cpu_utilization;
  }

  std::string statistics_csv = "";
  std::string means_string = "[";
  std::string stdevs_string = "[";
  std::string ideal_means_string = "[";
  std::string net_names = "[";
  std::string runtime_frames = "[";
  std::string runtime_ms = "[";

  size_t fractional_period = overall_period_load / 4;

  size_t stdevs_sum = 0;
  size_t i = 0;
  for (auto &&netDesc : nets){
    std::string single_net_runtime_frames_str = "[";
    std::string single_net_runtime_ms_str = "[";

    int j = 0;
    for (size_t frame_factor : netDesc->runtime_frames){
      if (j > 0){
        single_net_runtime_frames_str += ", ";
      }
      j++;
      single_net_runtime_frames_str += (std::to_string(frame_factor));
    }
    j = 0;
    for (size_t runtime : netDesc->runtime_ms){
      if (j > 0){
        single_net_runtime_ms_str += ", ";
      }
      j++;
      single_net_runtime_ms_str += (std::to_string(runtime));
    }
    single_net_runtime_frames_str += "]";
    single_net_runtime_ms_str += "]";

    // Calculate average and stdev real period in ms
    double sum_ms = std::accumulate(netDesc->runtime_ms.begin(),
                                 netDesc->runtime_ms.end(), 0.0);
    double mean_ms = sum_ms / netDesc->runtime_ms.size();

    double sq_sum_ms = std::inner_product(netDesc->runtime_ms.begin(),
                                       netDesc->runtime_ms.end(),
                                       netDesc->runtime_ms.begin(), 0.0);
    double stdev_ms = std::sqrt(sq_sum_ms / netDesc->runtime_ms.size()
                             - mean_ms * mean_ms);

    if (i>0){
      means_string += ", ";
      stdevs_string += ", ";
      ideal_means_string += ", ";
      net_names += ", ";
      runtime_frames += ",";
      runtime_ms += ", ";
    }
    means_string += std::to_string(mean_ms);
    stdevs_string += std::to_string(stdev_ms);
    stdevs_sum += stdev_ms;
    ideal_means_string += std::to_string(netDesc->period * fractional_period);
    net_names += "\"" + netDesc->name +
                 "\\nperiod=" + std::to_string(netDesc->period) +
                 "\\ncriticality=" + std::to_string(netDesc->criticality) + "\"";
    runtime_frames += single_net_runtime_frames_str;
    runtime_ms += single_net_runtime_ms_str;
    i++;
  }

  means_string += "]\n";
  stdevs_string += "]\n";
  ideal_means_string += "]\n";
  net_names += "]\n";
  runtime_frames += "]\n";
  runtime_ms += "]\n";

  LOG(INFO) << "STDEV_SUM = " << std::to_string(stdevs_sum) << "\n";
  LOG(INFO) << "STDEVS = " << stdevs_string << " STDEV_SUM==0 "
            << std::to_string(stdevs_sum==0) << "\n";

  std::ofstream stats;
  stats.open ("./seed" +
                 std::to_string(random_seed) + "-" +
                 "wcet" +
                 scheduling_policy +
                 ".py", std::ios::out | std::ios::app);

  stats << "### " << "Random Seed " << std::to_string(random_seed) << " ###\n";
  stats << "utilization=" << cpu_utilization << "\n";
  stats << "wcet" << scheduling_policy << "_means=" << means_string;
  stats << "wcet" << scheduling_policy << "_stdevs=" << stdevs_string;

  if (scheduling_policy == "RDTS"){
    stats << "ideal_means=" << ideal_means_string;
    stats << "net_names=" << net_names;
    stats << "random_seed=" << std::to_string(random_seed) << "\n";
  }

  stats << "runtime_frames=" << runtime_frames;
  stats << "runtime_ms=" << runtime_ms;
  stats.close();

  return 0;
}
