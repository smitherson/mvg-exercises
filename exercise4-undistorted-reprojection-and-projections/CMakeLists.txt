# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(opencv_example_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "-----------------------")
message(STATUS "CMAKE_CURRENT_LIST_FILE ${CMAKE_CURRENT_LIST_FILE}")

message(STATUS "CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}")

message(STATUS "CMAKE_CURRENT_LIST_LINE ${CMAKE_CURRENT_LIST_LINE}")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")

message(STATUS "PROJECT_BINARY_DIR ${PROJECT_BINARY_DIR}")
message(STATUS "PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}")

message(STATUS "CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}")


message(STATUS "PATH ${PATH}")
message(STATUS "$ ENV{PATH} $ENV{PATH}")

message(STATUS "CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")

message(STATUS "CMAKE_GENERATOR ${CMAKE_GENERATOR}")

message(STATUS "CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}")


message(STATUS "-----------------------")


message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -g -O0 -fprofile-arcs -ftest-coverage")

option(REPORT_COVERAGE            "Generate coverage report" ON)

if(REPORT_COVERAGE)
    if(NOT (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX) AND (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang"))
        message(FATAL_ERROR "Coverage needs GCC compiler. The current compiler ${CMAKE_C_COMPILER_ID} is not GNU gcc!")
    endif()

    if(NOT CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        message(FATAL_ERROR "Coverage: Code coverage results with an optimised (non-Debug) build may be misleading! Add -DCMAKE_BUILD_TYPE=Debug")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -g -O0 -fprofile-arcs -ftest-coverage")
endif(REPORT_COVERAGE)


if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from your sources
add_executable(radial-distortion radial-distortion.cpp)
#add_executable(3dto2d radial-distortion.cpp)

# Link your application with OpenCV libraries
target_link_libraries(radial-distortion ${OpenCV_LIBS})

