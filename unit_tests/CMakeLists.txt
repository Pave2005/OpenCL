cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")
add_definitions(-DTEST_DATA_DIR=\"${TEST_DATA_DIR}\")

find_package(GTest REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(tests
               main.cpp
			   tests.cpp)

target_compile_options(tests PRIVATE -O2)

target_link_libraries(tests
					  GTest::GTest
					  GTest::Main
					  ${OpenCL_LIBRARIES})

target_include_directories(tests PRIVATE
	                       ${CMAKE_SOURCE_DIR}/include
                           ${CMAKE_SOURCE_DIR}/unit_tests
						   ${OpenCL_INCLUDE_DIRS})

enable_testing()
add_test(NAME Tests COMMAND tests)
