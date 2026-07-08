cmake_minimum_required(VERSION 3.22)
project(sdxrn CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Public SDX header (dsp_runtime_c.h). Point SDX_INCLUDE_DIR at the include/
# directory of an unpacked SDK package (sdx-runtime-<platform>-<variant>.zip)
# or Android .aar headers/ dir; defaults to a sibling deeplearning4j checkout.
if(NOT DEFINED SDX_INCLUDE_DIR)
    set(SDX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../deeplearning4j/libnd4j/include")
endif()

# The SDX runtime shared library for the target ABI, packaged with the app
# under src/main/jniLibs/<abi>/ (libsdx_cpu.so from the SDK's Android package,
# or the monolithic libnd4jcpu.so — both export the same sdx* ABI).
set(JNILIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")
if(EXISTS "${JNILIBS_DIR}/libsdx_cpu.so")
    set(SDX_RUNTIME_SO "${JNILIBS_DIR}/libsdx_cpu.so")
elseif(EXISTS "${JNILIBS_DIR}/libnd4jcpu.so")
    set(SDX_RUNTIME_SO "${JNILIBS_DIR}/libnd4jcpu.so")
else()
    message(FATAL_ERROR
        "No SDX runtime library found in ${JNILIBS_DIR}. Copy libsdx_cpu.so "
        "(or libnd4jcpu.so) from the SDK's Android package into jniLibs/${ANDROID_ABI}/.")
endif()

add_library(sdx_runtime SHARED IMPORTED)
set_target_properties(sdx_runtime PROPERTIES IMPORTED_LOCATION "${SDX_RUNTIME_SO}")

add_library(sdxrn SHARED sdx_jni.cpp)
target_include_directories(sdxrn PRIVATE "${SDX_INCLUDE_DIR}")
target_link_libraries(sdxrn PRIVATE sdx_runtime)
