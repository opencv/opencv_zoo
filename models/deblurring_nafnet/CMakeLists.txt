cmake_minimum_required(VERSION 3.22.2)
project(opencv_zoo_deblurring_nafnet)

set(OPENCV_VERSION "5.0.0")
set(OPENCV_INSTALLATION_PATH "" CACHE PATH "Where to look for OpenCV installation")

# Find OpenCV
find_package(OpenCV ${OPENCV_VERSION} REQUIRED HINTS ${OPENCV_INSTALLATION_PATH})

add_executable(opencv_zoo_deblurring_nafnet demo.cpp)
target_link_libraries(opencv_zoo_deblurring_nafnet ${OpenCV_LIBS})
