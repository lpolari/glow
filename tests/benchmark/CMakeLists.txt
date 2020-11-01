if(GLOW_WITH_CPU AND NOT MSVC)
add_executable(AddBench
               AddBench.cpp)
target_link_libraries(AddBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        HostManager
                        CPURuntimeNative)

add_executable(SLSBench
               SLSBench.cpp)
target_link_libraries(SLSBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        HostManager
                        CPURuntimeNative)

add_executable(ConvBench
               ConvBench.cpp)
target_link_libraries(ConvBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        CPURuntimeNative)

add_executable(GemmBench
               GemmBench.cpp)
target_link_libraries(GemmBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(Int8GemmBench
               Int8GemmBench.cpp)
target_link_libraries(Int8GemmBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(GemmParallelBench
               GemmParallelBench.cpp)
target_link_libraries(GemmParallelBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(Int8GemmParallelBench
               Int8GemmParallelBench.cpp)
target_link_libraries(Int8GemmParallelBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(BatchGemmBench
               BatchGemmBench.cpp)
target_link_libraries(BatchGemmBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(TransposeBench
               TransposeBench.cpp)
target_link_libraries(TransposeBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(BERTProxyLayerBench
               BERTProxyLayerBench.cpp)
target_link_libraries(BERTProxyLayerBench
                      PRIVATE
                        Backends
                        ExecutionEngine
                        Graph
                        GraphOptimizer
                        HostManager
                        CPURuntimeNative)

add_executable(RuntimeBench
               RuntimeBench.cpp)
target_include_directories(RuntimeBench
                           PRIVATE
                             Backends
                             ${GLOW_SOURCE_DIR}/lib/Backends/CPU
                             ${GLOW_SOURCE_DIR}/lib/Backends/Interpreter)
target_link_libraries(RuntimeBench
                      PRIVATE
                        Backend
                        Backends
                        ExecutionEngine
                        Executor
                        HostManager
                        Graph
                        GraphOptimizer
                        benchmark)

add_executable(adas-pb
        adas-pb.cpp
        ../../tools/loader/Loader.cpp)
target_include_directories(adas-pb
        PRIVATE
        Backends
        ${GLOW_SOURCE_DIR}/lib/Backends/CPU
        ${GLOW_SOURCE_DIR}/lib/Backends/Interpreter
        ${GLOW_SOURCE_DIR}/tools/loader)
target_link_libraries(adas-pb
        PRIVATE
        Backend
        Backends
        ExecutionEngine
        Executor
        HostManager
        Graph
        GraphOptimizer
        benchmark
        Base
        Importer
        GraphOptimizer
        Quantization
        gtest
        nlohmann_json
        TestMain)
add_glow_test(NAME adas-pb
        COMMAND ${GLOW_BINARY_DIR}/tests/adas-pb
        -m=model
        --gtest_output=xml:adas-pb.xml
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
