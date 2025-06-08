# --- Project Setup ---
cmake_minimum_required(VERSION 3.15)
project(NNFSv2)

enable_testing()
set(CTEST_OUTPUT_ON_FAILURE 1)


# --- Source and Include Directories ---
set(PROJECT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

# --- Catch2 Setup ---
set(CATCH2_HEADER ${PROJECT_INCLUDE_DIR}/catch_amalgamated.hpp)
set(CATCH2_SOURCE ${PROJECT_SRC_DIR}/catch_amalgamated.cpp)

# --- Tensor Tests ---
file(GLOB_RECURSE TENSOR_SRC_SOURCES "${PROJECT_SRC_DIR}/tensor/*.cpp")
file(GLOB_RECURSE TENSOR_TEST_SOURCES "${PROJECT_TEST_DIR}/tensor/*.cpp")

add_executable(TensorTests
    ${TENSOR_TEST_SOURCES}
    ${TENSOR_SRC_SOURCES}
    ${CATCH2_SOURCE}
)

target_include_directories(TensorTests PRIVATE
    ${PROJECT_INCLUDE_DIR}
)

add_test(NAME TensorTests COMMAND TensorTests)

