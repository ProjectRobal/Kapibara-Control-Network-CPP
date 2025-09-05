cmake_minimum_required(VERSION 3.22)

project(KAC)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -g -Wall -ffast-math -O3 -std=c++23 -Wl,-z,stack-size=16777216")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -g -Wall  -O3 -ffast-math -std=c++23 -Wl,-z,stack-size=16777216")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/sources/")

include_directories(${PROTOBUF_INCLUDE_DIR})

find_package(OpenSSL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/sources/*.cpp"
        "${PROJECT_SOURCE_DIR}/sources/*.c"
        )

add_executable(main main.cpp ${all_SRCS})
target_include_directories(main PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(main PRIVATE OpenSSL::Crypto ${OpenCV_LIBS})