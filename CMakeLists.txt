cmake_minimum_required(VERSION 3.16)
project(ditserve VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(OpenSSL REQUIRED)

set(JSON_CACHE "${CMAKE_SOURCE_DIR}/.cache/deps/nlohmann3.11.3")
set(JSON_HEADER "${JSON_CACHE}/nlohmann/json.hpp")
set(JSON_SHA256 "9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6")
if(NOT EXISTS "${JSON_HEADER}")
    file(MAKE_DIRECTORY "${JSON_CACHE}/nlohmann")
    file(DOWNLOAD "https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp"
         "${JSON_HEADER}" EXPECTED_HASH "SHA256=${JSON_SHA256}" TLS_VERIFY ON STATUS JSON_DOWNLOAD_STATUS)
    list(GET JSON_DOWNLOAD_STATUS 0 JSON_DOWNLOAD_CODE)
    if(NOT JSON_DOWNLOAD_CODE EQUAL 0)
        file(REMOVE "${JSON_HEADER}")
        message(FATAL_ERROR "Cannot download pinned nlohmann JSON: ${JSON_DOWNLOAD_STATUS}")
    endif()
endif()
file(SHA256 "${JSON_HEADER}" JSON_ACTUAL_SHA256)
if(NOT JSON_ACTUAL_SHA256 STREQUAL JSON_SHA256)
    message(FATAL_ERROR "Cached nlohmann JSON 3.11.3 checksum mismatch")
endif()

find_package(Threads REQUIRED)
include(CTest)
add_library(ditserve_core
    src/ditserve/types.cpp
    src/ditserve/runtime/random.cpp src/ditserve/runtime/process.cpp
    src/ditserve/control/budget.cpp src/ditserve/control/dispatch.cpp src/ditserve/control/cluster.cpp
    src/ditserve/control/environment.cpp)
target_include_directories(ditserve_core PUBLIC include)
target_include_directories(ditserve_core SYSTEM PUBLIC "${JSON_CACHE}")
target_link_libraries(ditserve_core PUBLIC Threads::Threads)
target_compile_options(ditserve_core PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

add_subdirectory(benchmarks)
add_library(ditserve_application
    src/ditserve/control/study.cpp src/ditserve/interface/command.cpp)
target_link_libraries(ditserve_application PUBLIC ditserve_core ditserve_benchmark PRIVATE OpenSSL::Crypto)
target_compile_options(ditserve_application PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
target_compile_definitions(ditserve_application PRIVATE DITSERVE_PROJECT_ROOT="${CMAKE_SOURCE_DIR}" DITSERVE_VERSION="${PROJECT_VERSION}")
add_executable(ditserve cmd/main.cpp)
target_link_libraries(ditserve PRIVATE ditserve_application)
add_custom_command(TARGET ditserve POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E remove -f "$<TARGET_FILE_DIR:ditserve>/ditserve-core"
    COMMAND "${CMAKE_COMMAND}" -E create_symlink "$<TARGET_FILE_NAME:ditserve>" "$<TARGET_FILE_DIR:ditserve>/ditserve-core")
add_custom_target(ditserve-core DEPENDS ditserve)
install(TARGETS ditserve RUNTIME DESTINATION bin)
install(DIRECTORY workers/ditserve DESTINATION share/ditserve/python FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY include src cmd benchmarks DESTINATION share/ditserve
    FILES_MATCHING PATTERN "*.cpp" PATTERN "*.hpp" PATTERN "CMakeLists.txt" PATTERN "__pycache__" EXCLUDE)
install(FILES CMakeLists.txt DESTINATION share/ditserve)
install(DIRECTORY configs DESTINATION share/ditserve)

if(BUILD_TESTING)
    add_executable(supervision-tests tests/core/test_supervision.cpp)
    target_link_libraries(supervision-tests PRIVATE ditserve_core)
    add_test(NAME supervision-contracts COMMAND supervision-tests)
    add_executable(control-tests tests/core/test_control.cpp)
    target_link_libraries(control-tests PRIVATE ditserve_core)
    add_test(NAME control-contracts COMMAND control-tests)
    add_executable(environment-tests tests/core/test_environment.cpp)
    target_link_libraries(environment-tests PRIVATE ditserve_core)
    add_test(NAME environment-contracts COMMAND environment-tests)
    add_executable(command-tests tests/core/test_command.cpp)
    target_link_libraries(command-tests PRIVATE ditserve_application)
    add_test(NAME command-contracts COMMAND command-tests)
endif()
