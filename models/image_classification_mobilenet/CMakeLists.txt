cmake_minimum_required(VERSION 3.2)
set(project_name "mobilenet")

PROJECT (${project_name})

find_package(OpenCV REQUIRED)
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI


file(GLOB SourceFile
    "mobilenet.cpp")
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
