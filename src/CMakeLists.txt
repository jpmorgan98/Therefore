
# based off of: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/gesv/CMakeLists.txt

# ---[ Check cmake version.
cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)


# ---[ Project specification.
project(cusolver_examples LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

# ##########################################
# cusolver_examples build mode
# ##########################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# cusolver_examples dependencies
# ##########################################

#include_directories("${CUDA_INCLUDE_DIRS}")

# ##########################################
# cusolver_examples building flags
# ##########################################

# Global CXX/CUDA flags

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo")

# ##########################################
# cusolver_examples target
# ##########################################
include(cusolver_example.cmake)

#include_directories("${CMAKE_SOURCE_DIR}/../utils")

# Iterative refinement solver is added in CUDA 10.2
if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.2")
    add_cusolver_example(cusolver_examples "cusolver_axb" cusolver_axb.cu)
    #if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
    #    add_cusolver_example(cusolver_examples "cusolver_irs_expert" cusolver_irs_expert_cuda-11.cu)
    #else()
    #    add_cusolver_example(cusolver_examples "cusolver_irs_expert" cusolver_irs_expert_cuda-10.2.cu)
    #endif()
else()
    message("GESV solver routines are introduced in CUDA 10.2, update toolkit to get GESV functionality in cuSOLVER")
endif()

# ##########################################
# cusolver_examples directories
# ##########################################

# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Installation directories
set(CUSOLVER_EXAMPLES_BINARY_INSTALL_DIR "cusolver_examples/bin")

# ##########################################
# Install examples
# ##########################################

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF()