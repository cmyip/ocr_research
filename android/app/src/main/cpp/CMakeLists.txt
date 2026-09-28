cmake_minimum_required(VERSION 3.22.1)
project(axioslpr LANGUAGES C CXX)

# MNN runtime, pinned to the same version as the Python build that loaded these models.
# Only the C++ inference library is built; none of the MNN Python/CLI tools are involved.
include(FetchContent)
set(MNN_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "" FORCE)        # keep OpenCL/Vulkan backends inside libMNN.so
set(MNN_OPENCL ON CACHE BOOL "" FORCE)
set(MNN_VULKAN ON CACHE BOOL "" FORCE)
set(MNN_ARM82 ON CACHE BOOL "" FORCE)             # fp16 kernels for "Low" precision
set(MNN_USE_LOGCAT ON CACHE BOOL "" FORCE)
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
set(MNN_PORTABLE_BUILD ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    mnn
    URL https://github.com/alibaba/MNN/archive/refs/tags/3.6.1.tar.gz
)
FetchContent_MakeAvailable(mnn)

add_library(axioslpr SHARED mnn_jni.cpp)
target_include_directories(axioslpr PRIVATE ${mnn_SOURCE_DIR}/include)
target_link_libraries(axioslpr MNN log android)
