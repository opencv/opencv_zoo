cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 11)
set(project_name "opencv_zoo_face_expression_recognition")

PROJECT (${project_name})

set(OPENCV_VERSION "4.10.0")
set(OPENCV_INSTALLATION_PATH "" CACHE PATH "Where to look for OpenCV installation")
find_package(OpenCV ${OPENCV_VERSION} REQUIRED HINTS ${OPENCV_INSTALLATION_PATH})
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI

file(GLOB SourceFile
    "demo.cpp")
# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(${project_name} ${SourceFile})

# Link your application with OpenCV libraries
target_link_libraries(${project_name} PRIVATE ${OpenCV_LIBS})
