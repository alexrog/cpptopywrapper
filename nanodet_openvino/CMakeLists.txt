cmake_minimum_required(VERSION 3.4...3.18)
project(nanodet_openvino)

find_package(InferenceEngine REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ngraph REQUIRED)
find_package(realsense2 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)


add_subdirectory(pybind11)

# taken from https://stackoverflow.com/questions/60917800/how-to-get-the-opencv-image-from-python-and-use-it-in-c-in-pybind11/66549033#66549033
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "NUMPY_INCLUDE: " ${NUMPY_INCLUDE})
include_directories(${NUMPY_INCLUDE})

pybind11_add_module(nanodet_openvino src/main.cpp src/nanodet_openvino.cpp src/ndarray_converter.cpp)

target_link_libraries(nanodet_openvino PRIVATE ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES} ${OpenCV_LIBS} ${realsense2_LIBRARY} Threads::Threads)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.

target_compile_definitions(nanodet_openvino
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO}
)