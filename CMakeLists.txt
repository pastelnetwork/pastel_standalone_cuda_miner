cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set the CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75;86)

project(pastel_miner VERSION 1.0 LANGUAGES CXX CUDA)
project(pastel_miner_gtest VERSION 1.0 LANGUAGES CXX CUDA)

enable_language(CUDA)

# Add "cmake" subdirectory to the search path for CMake modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# libevent
find_package(LibEvent REQUIRED)
# GTest
find_package(GTest REQUIRED)

# Specify the executable and its source files
add_executable(pastel_miner)
add_executable(pastel_miner_gtest)

file(GLOB SRC_UTILS     "src/utils/*.cpp")
file(GLOB SRC_EQUIHASH  "src/equihash/*.cu" "src/equihash/*.cpp")
file(GLOB SRC_STRATUM   "src/stratum/*.cpp")
file(GLOB SRC_CUDA      "src/cuda/*.cu")
file(GLOB SRC_MAIN      "src/main.cpp")

file(GLOB SRC_GTEST     "src/gtest/*.cpp")

target_sources(pastel_miner
	PRIVATE
		${SRC_UTILS}
		${SRC_EQUIHASH}
		${SRC_STRATUM}
		${SRC_CUDA}
		${SRC_MAIN}
)

target_sources(pastel_miner_gtest
	PRIVATE
		${SRC_UTILS}
		${SRC_EQUIHASH}
		${SRC_STRATUM}
		${SRC_CUDA}
		${SRC_GTEST}
)

if (LibEvent_FOUND)
	target_include_directories(pastel_miner PRIVATE ${LIBEVENT_INCLUDE_DIR})
	target_link_libraries(pastel_miner ${LIBEVENT_LIBRARIES})

	target_include_directories(pastel_miner_gtest PRIVATE ${LIBEVENT_INCLUDE_DIR})
	target_link_libraries(pastel_miner_gtest ${LIBEVENT_LIBRARIES})
endif() 

if (GTest_FOUND)
	target_include_directories(pastel_miner_gtest PRIVATE ${GTEST_INCLUDE_DIRS})
	target_link_libraries(pastel_miner_gtest ${GTEST_LIBRARIES})
endif()

target_include_directories(pastel_miner PRIVATE include/ ${CMAKE_SOURCE_DIR})
target_include_directories(pastel_miner_gtest PRIVATE include/ ${CMAKE_SOURCE_DIR})

# Set target properties, if needed (e.g., C++ standard, CUDA architectures)
set_target_properties(pastel_miner PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CXX_STANDARD 17
	CUDA_STANDARD 17)

set_target_properties(pastel_miner_gtest PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CXX_STANDARD 17
	CUDA_STANDARD 17)