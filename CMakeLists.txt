cmake_minimum_required(VERSION 3.14)

project(BitonicSort)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(GTest REQUIRED)
find_package(OpenCL REQUIRED)

enable_testing()
add_subdirectory(unit_tests)

add_executable(bsort ./main.cpp)

target_include_directories(bsort PRIVATE
	                       ${CMAKE_SOURCE_DIR}/include
	                       ${CMAKE_SOURCE_DIR}/utils
						   ${CMAKE_SOURCE_DIR}/config
	                       ${OpenCL_INCLUDE_DIRS})

target_link_libraries(bsort PRIVATE ${OpenCL_LIBRARIES})

target_compile_options(bsort PRIVATE -O2)
