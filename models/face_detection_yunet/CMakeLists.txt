cmake_minimum_required(VERSION 3.24.0)
project(opencv_zoo_face_detection_yunet)

set(OPENCV_VERSION "4.7.0")
set(OPENCV_INSTALLATION_PATH "" CACHE PATH "Where to look for OpenCV installation")

# Find OpenCV
find_package(OpenCV ${OPENCV_VERSION} REQUIRED HINTS ${OPENCV_INSTALLATION_PATH})

add_executable(demo demo.cpp)
target_link_libraries(demo ${OpenCV_LIBS})
