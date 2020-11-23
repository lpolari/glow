# ADAS DNN Toolchain


## Getting Started

The following introductions describe the installation and usage of
the adas-dnn-toolchain that has been developed in the context of my 
master thesis "Scheduling von Deep Learning-Netzen mit Echtzeitanforderungen
in Fahrassistenzsystemen"

### System Requirements

The toolchain has been tested on Ubuntu 18.04.

#### Get the source and data repositories!

  ```bash
  mkdir adas-dnn-toolchain
  cd adas-dnn-toolchain
  mkdir -p data/adas-pb-output/partition_configs
  git clone https://github.com/lpolari/glow.git src/
  cd src/
  ```

#### Submodules

Glow depends on a few submodules: googletest, onnx, and a library
for FP16 conversions.

To get them, from the glow directory, run:

  ```bash
  git submodule update --init --recursive
  ```

#### Source dependencies

Glow depends on `fmt`, which must be built from source:
```bash
git clone https://github.com/fmtlib/fmt
mkdir fmt/build
cd fmt/build
cmake ..
make
sudo make install
cd ../../..
```


#### Ubuntu

In order to build Glow on Ubuntu it is necessary to install a few packages. The
following command should install the required dependencies:

  ```bash
  sudo apt-get install clang clang-8 cmake graphviz libpng-dev \
      libprotobuf-dev llvm-8 llvm-8-dev ninja-build protobuf-compiler wget \
      opencl-headers libgoogle-glog-dev libboost-all-dev \
      libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev \
      libjemalloc-dev libpthread-stubs0-dev sqlite3
  ```

[Note: Ubuntu 16.04 and 18.04 ship with llvm-6 and need to be upgraded before building Glow. Building Glow on Ubuntu 16.04 with llvm-7 fails because llvm-7 xenial distribution uses an older c++ ABI, however building Glow on Ubuntu 18.04 with llvm-7 has been tested and is successful]

It may be desirable to use `update-alternatives` to manage the version of
clang/clang++:

  ```bash
  sudo update-alternatives --install /usr/bin/clang clang \
      /usr/lib/llvm-8/bin/clang 50
  sudo update-alternatives --install /usr/bin/clang++ clang++ \
      /usr/lib/llvm-8/bin/clang++ 50
  ```

Glow uses the system default C/C++ compiler (/usr/bin/c++), and so you may also
want to switch your default C/C++ compiler to clang:

  ```bash
  sudo update-alternatives --config cc
      # Select the option corresponding to /usr/bin/clang ...
  sudo update-alternatives --config c++
      # Select the option corresponding to /usr/bin/clang++ ...
  ```

Glow *should* build just fine with gcc (e.g. gcc 5.4), but we mostly use clang
and are more attentive to compatibility with clang.

Finally, in order to support the ONNX net serialization format, Glow requires
`protobuf >= 2.6.1`, but the above command may install older
version on older Ubuntu (e.g. 14.04). If this is the case, we suggest to look
at `utils/install_protobuf.sh` to install a newer version from source.

For details on installing OpenCL on Ubuntu please see
[these instructions](docs/Building.md#opencl-on-ubuntu).

### Configure and Build

To build the compiler, create a build directory and run cmake on the source
directory. It's a good idea to build two configurations (Release and Debug)
because some programs take a really long time to run in Debug mode. It's also a
good idea to build the project outside of the source directory.

  ```bash
  cd src
  git checkout lpolari/feature-adas-dnnr
  cd ..
  mkdir adas-dnnr
  cd adas-dnnr
  cmake -G Ninja ../src -DCMAKE_BUILD_TYPE=Release
  ninja adas-dnnr

  cd ../src
  git checkout lpolari/feature-adas-pb
  cd ..
  mkdir adas-pb
  cd adas-pb
  cmake -G Ninja ../src -DCMAKE_BUILD_TYPE=Release
  ninja adas-pb

  cd ../src
  git checkout lpolari/feature-adas-lrm
  cd ../
  mkdir adas-lrm
  cd adas-lrm
  cmake -G Ninja ../src -DCMAKE_BUILD_TYPE=Release
  ninja adas-lrm
  ```

### Download ONNX models
  Use the utility script in the glow repository to download
  the ONNX models. To download all models that has been
  used for the tests described in the thesis use the
  following command.
  ```bash
  cd ../data
  mkdir onnx-models
  cd onnx-models
  python ../../src/utils/download_datasets_and_models.py -c {resnet50,shufflenet,inception_v1,inception_v2}
  ```

### Download Image data

The image data used to for the ONNX models were taken
from the imageNet challenge 2012. The download requires
registration on the image-net website. The images shall
be stored in the data/images/imagenet directory to be 
recognized by the adas-dnnr-tool. Visit the following 
link for further information:

http://www.image-net.org/challenges/LSVRC/2012/

### Create layer runtime measurements database

Create and load the sqlite file
```bash
cd ..
sqlite3 layer-runtime-measurements.db
```
    
Create the traces table
```sqlite
CREATE TABLE traces(
id INTEGER PRIMARY KEY AUTOINCREMENT,
run_id INTEGER NOT NULL,
network_name TEXT NOT NULL,
node_name TEXT NOT NULL,
duration INTEGER NOT NULL);
```

### Fill the layer runtime measurements database

 ```bash
  cd ../adas-lrm
  export NET_NAME=<net_name>
  export NET_INPUT_NAME=<input_name>
  ./bin/adas-lrm -auto-instrument -trace-path=<path>
  ```

### Create partition configs for a networks

The parameters have to be specified as enviroment variables.
The partition-config of the network and period is saved into
the data-directory where it is located automatically by the 
adas-dnnr-tool.

 ```bash
  cd ../adas-pb
  export NET_NAME=<net_name>
  export NET_INPUT_NAME=<input_name>
  export NET_PERIOD=<period>
  ./tests/adas-pb --benchmark_repetitions=1000 --benchmark_min_time=0.01
```

### Start a test run for the adas DNN runtime enviroment

The adas-dnnr tool can be provided with a random seed. Based on that seed
4 networks are chosen for execution with random periods and criticalities.
Furthermore the scheduling policy has to be specified (either RDTS or MCDTS).
In the default configuration the tool chooses between resnet50, shufflenet,
inception_v1 and inception_v2 networks, with periods of 1, 2 and 4 and 2
different criticality levels. To start a random a seed has to be provided
and run the corresponding partition_configs have to be generated for all 
networks and periods. The tracing output can be visualised using Google 
Chrome's Trace-Viewer.

```bash
  cd ../adas-dnnr
  sudo ./bin/adas-dnnr -auto-instrument -glow_partitioner_enable_json_config -random-seed=<seed> -trace-path=<path> -scheduling="MCDTS"
```


It's possible to configure and build the compiler with any CMake generator,
like GNU Makefiles, Ninja and Xcode build.

For platform-specific build instructions and advanced options, such as
building with Address-Sanitizers refer to this guide:
[Building the Compiler](docs/Building.md).


#### Building with dependencies (LLVM)

By default, Glow will use a system provided LLVM.  Note that Glow requires LLVM
7.0 or later. If you have LLVM installed in a non-default location (for
example, if you installed it using Homebrew on macOS), you need to tell CMake
where to find llvm using `-DLLVM_DIR`. For example, if LLVM were
installed in `/usr/local/opt`:

  ```bash
  cmake -G Ninja ../src \
      -DCMAKE_BUILD_TYPE=Debug \
      -DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm
  ```

If LLVM is not available on your system you'll need to build it manually.  Run
the script '`/utils/build_llvm.sh` to clone, build and install LLVM in a local
directory. You will need to configure Glow with the flag `-DLLVM_DIR` to tell
the build system where to find LLVM given the local directory you installed it
in (e.g. `-DLLVM_DIR=/path/to/llvm_install/lib/cmake/llvm` if using
`build_llvm.sh`).


## License

Glow is licensed under the [Apache 2.0 License](LICENSE).
