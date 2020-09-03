cmake_minimum_required(VERSION 2.8)

project (gomoku)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(Threads)

include_directories(src)

add_library(gomoku SHARED src/library.cpp src/gomoku.cpp)
target_link_libraries(gomoku ${CMAKE_THREAD_LIBS_INIT})


if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=gnu++0x)
endif()

