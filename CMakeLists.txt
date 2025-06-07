cmake_minimum_required(VERSION 3.28)
project(NNFSv2)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h")

add_library(NNFSLibrary STATIC ${SOURCES})

add_executable(NNFSExample examples/main.cpp)
target_link_libraries(NNFSExample NNFSLibrary)

