cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(ExternalProject)

if (NOT (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Release"))
  message(FATAL_ERROR "CMAKE_BUILD_TYPE is not set. Please specify a build type (Debug or Release) using -DCMAKE_BUILD_TYPE=<type>.")
endif()

# Define global output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# Set the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 61;75;86)
# Ensure consistent runtime library usage
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")  # Use /MDd for Debug and /MD for Release


project(pastel_miner_gtest VERSION 1.0 LANGUAGES CXX CUDA)
project(pastel_miner VERSION 1.0 LANGUAGES CXX CUDA)

enable_language(CUDA)

# Add "cmake" subdirectory to the search path for CMake modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set installation directory for dependencies
set(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/depends")
set(DEPS_DIR_PACKAGES ${DEPS_DIR}/packages)
set(DEPS_DIR_RELEASE ${DEPS_DIR}/bin/release)
set(DEPS_DIR_DEBUG ${DEPS_DIR}/bin/debug)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEPS_INSTALL_PREFIX "${DEPS_DIR_DEBUG}")
    set(DEPS_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
else()
    set(DEPS_INSTALL_PREFIX "${DEPS_DIR_RELEASE}")
    set(DEPS_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

set(DEPS_DIR_INCLUDE "$<IF:$<CONFIG:Debug>,${DEPS_DIR_DEBUG}/include,${DEPS_DIR_RELEASE}/include>")
set(DEPS_DIR_LIB "$<IF:$<CONFIG:Debug>,${DEPS_DIR_DEBUG}/lib,${DEPS_DIR_RELEASE}/lib>")

set(DEPS_SUPERBUILD_CONFIGS "Debug;Release")

ExternalProject_Add(
	libevent
	PREFIX ${DEPS_DIR_PACKAGES}/libevent
	GIT_REPOSITORY https://github.com/libevent/libevent.git
	GIT_TAG release-2.1.12-stable
	CMAKE_ARGS
           -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_PREFIX}
           -DCMAKE_BUILD_TYPE=${CONFIG}
           -DCMAKE_MSVC_RUNTIME_LIBRARY=${DEPS_MSVC_RUNTIME_LIBRARY} # Use dynamic runtime for MSVC build
           -DEVENT__BUILD_SHARED_LIBRARIES=ON  # Build as a shared library (DLL)
           -DEVENT__DISABLE_THREAD_SUPPORT=OFF # Enable thread support
           -DEVENT__DISABLE_TESTS=ON           # Disable tests
           -DEVENT__DISABLE_OPENSSL=ON         # Disable OpenSSL support
           -DEVENT__DISABLE_BENCHMARK=ON       # Disable benchmarks
           -DEVENT__DISABLE_REGRESS=ON         # Disable the regress tests
           -DEVENT__DISABLE_SAMPLES=ON         # Disable sample files
	INSTALL_DIR ${DEPS_INSTALL_PREFIX}
)

ExternalProject_Add(
	libgtest
	PREFIX ${DEPS_DIR_PACKAGES}/gtest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG v1.15.2
	CMAKE_ARGS
	    -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_PREFIX}
	    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
	    -DCMAKE_MSVC_RUNTIME_LIBRARY=${DEPS_MSVC_RUNTIME_LIBRARY}
	    -DGTEST_BUILD_SAMPLES=OFF
           -Dgtest_force_shared_crt=ON
	INSTALL_DIR ${DEPS_INSTALL_PREFIX}
)

ExternalProject_Add(
    spdlog
    PREFIX ${DEPS_DIR_PACKAGES}/spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${DEPS_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DSPDLOG_BUILD_EXAMPLES=OFF
        -DSPDLOG_BUILD_TESTS=OFF
)

# Specify the executable and its source files
add_executable(pastel_miner)
add_executable(pastel_miner_gtest)

# source files
file(GLOB SRC_UTILS     "src/utils/*.cpp")
file(GLOB SRC_EQUIHASH  "src/equihash/*.cpp")
file(GLOB SRC_STRATUM   "src/stratum/*.cpp")
file(GLOB SRC_KERNEL    "src/kernel/*.cu")
file(GLOB SRC_MAIN      "src/main.cpp")
file(GLOB SRC_GTEST     "src/gtest/*.c*")

# header files
file(GLOB HDR_INCLUDE   "include/*.h")
file(GLOB HDR_COMPAT    "include/compat/*.h")
file(GLOB HDR_UTILS     "src/utils/*.h")
file(GLOB HDR_EQUIHASH  "src/equihash/*.h")
file(GLOB HDR_STRATUM   "src/stratum/*.h")
file(GLOB HDR_KERNEL    "src/kernel/*.h")

# Group source and header files into folders
source_group("include"         FILES ${HDR_INCLUDE})
source_group("include/compat"  FILES ${HDR_COMPAT})
source_group("utils"           FILES ${SRC_UTILS}    ${HDR_UTILS})
source_group("equihash"        FILES ${SRC_EQUIHASH} ${HDR_EQUIHASH})
source_group("stratum"         FILES ${SRC_STRATUM}  ${HDR_STRATUM})
source_group("kernel"          FILES ${SRC_KERNEL}   ${HDR_KERNEL})

# Visual Studio settings
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT pastel_miner)

target_sources(pastel_miner
	PRIVATE
		${HDR_INCLUDE}
		${HDR_COMPAT}
		${SRC_UTILS} ${HDR_UTILS}
		${SRC_EQUIHASH} ${HDR_EQUIHASH}
		${SRC_STRATUM} ${HDR_STRATUM}
		${SRC_KERNEL} ${HDR_KERNEL}
		${SRC_MAIN}
)

target_sources(pastel_miner_gtest
	PRIVATE
		${HDR_INCLUDE}
		${HDR_COMPAT}
		${SRC_UTILS} ${HDR_UTILS}
		${SRC_EQUIHASH} ${HDR_EQUIHASH}
		${SRC_STRATUM} ${HDR_STRATUM}
		${SRC_KERNEL} ${HDR_KERNEL}
		${SRC_GTEST}
)

# Dependencies
add_dependencies(pastel_miner libevent spdlog)
add_dependencies(pastel_miner_gtest libevent spdlog libgtest)

if (WIN32)
    set(LIBEVENT_LIB_NAMES "event.lib")
    set(GTEST_LIB_NAMES "gtest.lib")
    set(LIBEVENT_SHAREDLIB_NAME "event.dll")
elseif (UNIX)
    set(LIBEVENT_LIB_NAMES "event" "event_core" "event_pthreads")
    set(GTEST_LIB_NAMES "gtest")
else()
    message(FATAL_ERROR "Unsupported build platform")
endif()

target_include_directories(pastel_miner PRIVATE
    include/
    ${DEPS_DIR_INCLUDE}
    ${CMAKE_SOURCE_DIR})
target_link_directories(pastel_miner PRIVATE
    ${DEPS_DIR_LIB})
target_link_libraries(pastel_miner PRIVATE
    $<$<PLATFORM_ID:Windows>:Ws2_32.lib>
    ${LIBEVENT_LIB_NAMES})

target_include_directories(pastel_miner_gtest PRIVATE
    include/
    ${DEPS_DIR_INCLUDE}
    ${CMAKE_SOURCE_DIR})
target_link_directories(pastel_miner_gtest PRIVATE
    ${DEPS_DIR_LIB})
target_link_libraries(pastel_miner_gtest PRIVATE
    $<$<PLATFORM_ID:Windows>:Ws2_32.lib>
    ${LIBEVENT_LIB_NAMES}
    ${GTEST_LIB_NAMES})

if (WIN32)
    add_custom_command(
        TARGET pastel_miner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEPS_DIR_LIB}/${LIBEVENT_SHAREDLIB_NAME}"
            "$<TARGET_FILE_DIR:pastel_miner>"
        COMMENT "Copying libevent to the output directory"
    )

    add_custom_command(
        TARGET pastel_miner_gtest POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEPS_DIR_LIB}/${LIBEVENT_SHAREDLIB_NAME}"
            "$<TARGET_FILE_DIR:pastel_miner_gtest>"
        COMMENT "Copying libevent to the output directory"
    )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pastel_miner PRIVATE 
		"$<$<COMPILE_LANGUAGE:CUDA>:-G>")
    target_compile_options(pastel_miner_gtest PRIVATE 
		"$<$<COMPILE_LANGUAGE:CUDA>:-G>")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(pastel_miner PRIVATE
		"$<$<COMPILE_LANGUAGE:CUDA>:-O3>")
    target_compile_options(pastel_miner_gtest PRIVATE 
		"$<$<COMPILE_LANGUAGE:CUDA>:-O3>")
endif()

target_compile_definitions(pastel_miner PRIVATE
    HAVE_CONFIG_H)
target_compile_definitions(pastel_miner_gtest PRIVATE
    HAVE_CONFIG_H)

# Set target properties, if needed (e.g., C++ standard, CUDA architectures)
set_target_properties(pastel_miner PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CXX_STANDARD 20
	CUDA_STANDARD 17)

set_target_properties(pastel_miner_gtest PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CXX_STANDARD 20
	CUDA_STANDARD 17)
# Add verbose flag to CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --verbose")

set(CPACK_PACKAGE_NAME "PastelMiner")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_VENDOR "PastelNetwork")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
include(CPack)