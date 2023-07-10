cmake_minimum_required(VERSION 3.5)
project(yolo_realtime_inference_detect)


set(OpenCV_DIR F:\\cpp_tools\\tools\\opencv_4_5_5_release\\build)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(yolo_realtime_inference_detect yolo_realtime_inference_detect.cpp)

target_link_libraries(yolo_realtime_inference_detect PRIVATE ${OpenCV_LIBS})
