cmake_minimum_required(VERSION 3.12)
project(onnxruntime_inference)

# 设置 ONNX Runtime 的路径
set(ONNXRUNTIME_DIR F:\\cpp_tools\\tools\\onnxruntime_1_10_0)

# 添加源文件
add_executable(onnxruntime_inference onnxruntime_inference_hangtian.cpp)

# 添加 ONNX Runtime 的头文件路径
target_include_directories(onnxruntime_inference PRIVATE ${ONNXRUNTIME_DIR}/include)

# 添加 ONNX Runtime 的库文件路径
target_link_directories(onnxruntime_inference PRIVATE ${ONNXRUNTIME_DIR}/lib)

# 设置 OpenCV 的路径
set(OpenCV_DIR F:\\cpp_tools\\tools\\opencv_4_5_5_release\\build)

# 查找并添加 OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(onnxruntime_inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(onnxruntime_inference PRIVATE ${OpenCV_LIBS})

# 添加 ONNX Runtime 的链接库
target_link_libraries(onnxruntime_inference PRIVATE onnxruntime)

