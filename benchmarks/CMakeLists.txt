add_library(ditserve_benchmark
    src/backend.cpp src/cost.cpp src/metrics.cpp
    src/policy.cpp src/simulator.cpp src/workload.cpp)
target_include_directories(ditserve_benchmark PUBLIC include)
target_link_libraries(ditserve_benchmark PUBLIC ditserve_core PRIVATE OpenSSL::Crypto)
target_compile_options(ditserve_benchmark PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

if(BUILD_TESTING)
    add_executable(simulation-tests tests/test_simulator.cpp)
    target_link_libraries(simulation-tests PRIVATE ditserve_benchmark)
    target_compile_definitions(simulation-tests PRIVATE BASELINE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/python-core-baseline.json")
    add_test(NAME simulation-contracts COMMAND simulation-tests)
endif()
