cmake_minimum_required(VERSION 3.22.2)
project(opencv_zoo_edge_detection_dexined)

set(OPENCV_VERSION "5.0.0")
set(OPENCV_INSTALLATION_PATH "" CACHE PATH "Where to look for OpenCV installation")

# Find OpenCV
find_package(OpenCV ${OPENCV_VERSION} REQUIRED HINTS ${OPENCV_INSTALLATION_PATH})

add_executable(edge_detection edge_detection.cpp)
target_link_libraries(edge_detection ${OpenCV_LIBS})
