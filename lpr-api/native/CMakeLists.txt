cmake_minimum_required(VERSION 3.22)
project(lprmnn LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# MNN runtime, pinned to the version that loaded these models in the Python PoC and the Android
# app. Only the C++ inference library is built (CPU backend, static); none of the MNN Python/CLI
# tools are involved, so nothing here phones home (see HANDOVER.md).
include(FetchContent)
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "" FORCE)
set(MNN_METAL OFF CACHE BOOL "" FORCE)
set(MNN_OPENCL OFF CACHE BOOL "" FORCE)
set(MNN_VULKAN OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TRAIN OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_QUANTOOLS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_LLM OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_OPENCV OFF CACHE BOOL "" FORCE)
set(MNN_EVALUATION OFF CACHE BOOL "" FORCE)

if(DEFINED ENV{MNN_SOURCE_DIR})
    FetchContent_Declare(mnn SOURCE_DIR $ENV{MNN_SOURCE_DIR})
else()
    FetchContent_Declare(mnn URL https://github.com/alibaba/MNN/archive/refs/tags/3.6.1.tar.gz)
endif()
FetchContent_MakeAvailable(mnn)

add_library(lprmnn STATIC mnn_shim.cpp)
target_include_directories(lprmnn PRIVATE ${mnn_SOURCE_DIR}/include)
add_dependencies(lprmnn MNN)
