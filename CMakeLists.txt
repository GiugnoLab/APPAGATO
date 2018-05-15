set(PROJECT "APPAGATO")
project(${PROJECT} CXX)

cmake_minimum_required(VERSION 3.0)
FIND_PACKAGE(CUDA)

message("\nTest on cmake v" ${CMAKE_VERSION} "  System: " ${CMAKE_SYSTEM} "\n")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

if(CUDA_VERSION_MAJOR LESS 7)
	message(FATAL_ERROR "\nRequired Cuda Toolkit >= v7.0   Founded: Cuda Toolkit v${CUDA_VERSION_MAJOR}")
endif()

set(CMAKE_LEGACY_CYGWIN_WIN32 0)

# ------------------------------------------------------------------------------

set(XLIB_DIR    XLib)

include_directories(${PROJECT_SOURCE_DIR}/include/
                    ${PROJECT_SOURCE_DIR}/${XLIB_DIR}
                    ${PROJECT_SOURCE_DIR}/cub-1.4.1)

set(HOST_SRC        ${PROJECT_SOURCE_DIR}/src/Host/GraphAPP.cpp
                    ${PROJECT_SOURCE_DIR}/src/Host/approx_subgraph_iso.cpp
                    ${PROJECT_SOURCE_DIR}/src/Host/similarity.cpp
                    ${PROJECT_SOURCE_DIR}/src/Host/seed.cpp
                    ${PROJECT_SOURCE_DIR}/src/Host/extend.cpp)

set(DEVICE_SRC      ${PROJECT_SOURCE_DIR}/src/Device/approx_subgraph_iso.cu
                    ${PROJECT_SOURCE_DIR}/src/Device/similarity.cu
                    ${PROJECT_SOURCE_DIR}/src/Device/seed.cu
                    ${PROJECT_SOURCE_DIR}/src/Device/extend.cu)

set(HOST_UTIL_SRC   ${PROJECT_SOURCE_DIR}/${XLIB_DIR}/Base/Host/src/fUtil.cpp
                    ${PROJECT_SOURCE_DIR}/${XLIB_DIR}/Base/Host/src/Timer.cpp)

set(DEVICE_UTIL_SRC ${PROJECT_SOURCE_DIR}/${XLIB_DIR}/Base/Device/Util/src/cuda_util.cu
                    ${PROJECT_SOURCE_DIR}/${XLIB_DIR}/Base/Device/Util/src/Timer.cu)

# ------------------------------------------------------------------------------

SET(HOST_W_FLAGS "-Wall -Wextra -Wno-deprecated-declarations -Wno-missing-field-initializers")
SET(DEVICE_W_FLAGS -Wall,-Wextra,-Wno-deprecated-declarations,-Wno-missing-field-initializers)

if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    SET(HOST_W_FLAGS "${HOST_W_FLAGS} -fno-diagnostics-show-caret -ftrack-macro-expansion=0")
    SET(DEVICE_W_FLAGS "${DEVICE_W_FLAGS},-fno-diagnostics-show-caret,-ftrack-macro-expansion=0")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HOST_W_FLAGS}")

# ------------------------------------------------------------------------------

add_executable(SeqAppagato_info ${PROJECT_SOURCE_DIR}/Appagato.cpp
                    ${HOST_SRC} ${HOST_UTIL_SRC})
set_target_properties(SeqAppagato_info PROPERTIES COMPILE_FLAGS
                    "-DPRINT_INFO -DCHECK_ERROR -DGET_TIME")

add_executable(SeqAppagato_rel ${PROJECT_SOURCE_DIR}/Appagato.cpp
                   ${HOST_SRC} ${HOST_UTIL_SRC})
set_target_properties(SeqAppagato_rel PROPERTIES COMPILE_FLAGS "-DNDEBUG")

# ------------------------------------------------------------------------------

if (CUDA_FOUND)

set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "-std=c++11")

set(ARCH "30" CACHE STRING "architectures")

if (ARCH STREQUAL "30")
	message("\nGPU architecture: 3.0\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30" "-DARCH=300")
elseif (ARCH STREQUAL "32")
    message("\nGPU architecture: 3.2\n")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_32" "-DARCH=320")
elseif (ARCH STREQUAL "35")
	message("\nGPU architecture: 3.5\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35" "-DARCH=350")
elseif (ARCH STREQUAL "35")
    message("\nGPU architecture: 3.7\n")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_37" "-DARCH=370")
elseif (ARCH STREQUAL "50")
	message("\nGPU architecture: 5.0\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50" "-DARCH=500")
elseif (ARCH STREQUAL "52")
	message("\nGPU architecture: 5.2\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52" "-DARCH=520")
elseif (ARCH STREQUAL "53")
	message("\nGPU architecture: 5.3\n")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_53" "-DARCH=530")
else()
	message(FATAL_ERROR "\nplease select GPU architecture: \"cmake -DARCH=[ 30 | 32 | 35 | 37 | 50 | 52 | 53 ]\"")
endif()

# ------------------------------------------------------------------------------
if (CMAKE_BUILD_TYPE STREQUAL "")
	set (CMAKE_BUILD_TYPE Release)
endif()

message("Selected CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} "\n")
if (    (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
	AND (NOT CMAKE_BUILD_TYPE STREQUAL "Debug"))
	message(FATAL_ERROR "\nwrong BUILD_TYPE: \"-DCMAKE_BUILD_TYPE=[ Release | Debug ]\"")
endif()

#"-DDEBUG_READ"
#debug
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g" "-G")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

#"-use_fast_math"
#release options
list(APPEND CUDA_NVCC_FLAGS_RELEASE "-Xcompiler=-O3,-march=native,-funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")

# ------------------------------------------------------------------------------

cuda_add_executable(ParallelAppagato_info ${PROJECT_SOURCE_DIR}/Appagato.cpp
                    ${HOST_SRC} ${HOST_UTIL_SRC} ${DEVICE_SRC} ${DEVICE_UTIL_SRC} ${CUDA_curand_LIBRARY}
                    OPTIONS -DPRINT_INFO -DCHECK_ERROR -DCHECK_RESULT -DGET_TIME -D__DEVICE__ -DNDEBUG
                -Xcompiler=-DPRINT_INFO,-DCHECK_ERROR,-DCHECK_RESULT,-DGET_TIME,-D__DEVICE__,-DNDEBUG)
set_target_properties(ParallelAppagato_info PROPERTIES COMPILE_FLAGS
                "-DPRINT_INFO -DCHECK_ERROR -DCHECK_RESULT -DGET_TIME -D__DEVICE__ -DNDEBUG")


cuda_add_executable(ParallelAppagato_rel ${PROJECT_SOURCE_DIR}/Appagato.cpp
                    ${HOST_SRC} ${HOST_UTIL_SRC} ${DEVICE_SRC} ${DEVICE_UTIL_SRC} ${CUDA_curand_LIBRARY}
                    OPTIONS -DNDEBUG -D__DEVICE__ -DNDEBUG -Xcompiler=-DNDEBUG,-D__DEVICE__)
set_target_properties(ParallelAppagato_rel PROPERTIES COMPILE_FLAGS "-DNDEBUG -D__DEVICE__")

endif()
#-------------------------------------------------------------------------------

add_executable(Converter ${PROJECT_SOURCE_DIR}/Converter/Converter.cpp ${HOST_UTIL_SRC})

set(EXECUTABLE_OUTPUT_PATH .)

add_custom_target(clean-all rm -rf *)
