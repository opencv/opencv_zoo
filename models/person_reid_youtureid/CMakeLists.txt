cmake_minimum_required(VERSION 3.24.0)
project(opencv_zoo_person_reid_youtureid)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENCV_VERSION "4.10.0")
set(OPENCV_INSTALLATION_PATH "" CACHE PATH "Where to look for OpenCV installation")

find_package(OpenCV ${OPENCV_VERSION} REQUIRED HINTS ${OPENCV_INSTALLATION_PATH})

add_executable(demo demo.cpp)
target_link_libraries(demo ${OpenCV_LIBS} stdc++fs) 

