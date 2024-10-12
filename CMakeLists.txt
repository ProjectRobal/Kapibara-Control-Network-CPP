cmake_minimum_required(VERSION 3.22)

project(Kolejka)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -g -Wall -ffast-math -std=c++20 -Wl,-z,stack-size=16777216")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/sources/")

include_directories(${PROTOBUF_INCLUDE_DIR})

file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/sources/*.cpp"
        "${PROJECT_SOURCE_DIR}/sources/*.c"
        )

add_executable(main main.cpp ${all_SRCS})
target_include_directories(main PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(main PRIVATE)