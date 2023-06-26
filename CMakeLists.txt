cmake_minimum_required(VERSION 3.5)
project(yolov5n_inference_seg)


set(OpenCV_DIR F:\\cpp_tools\\tools\\opencv_4_5_5_release\\build)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(yolov5n_inference_seg yolov5n_inference_seg.cpp)

target_link_libraries(yolov5n_inference_seg PRIVATE ${OpenCV_LIBS})
