cmake_minimum_required(VERSION 3.12)
project(test)

# 设置 ONNX Runtime 的路径
# F:\\cpp_tools\\tools\\onnxruntime-win-x64-1.14.1
# F:\\cpp_tools\\tools\\onnxruntime-win-x64-1.10.0
set(ONNXRUNTIME_DIR F:\\cpp_tools\\tools\\onnxruntime-win-x64-1.10.0)

# 添加源文件
add_executable(test test.cpp)
# 执行得到可执行文件
# ADD_LIBRARY(test add_executable test.cpp)

# 添加 ONNX Runtime 的头文件路径
target_include_directories(test PRIVATE ${ONNXRUNTIME_DIR}/include)

# 添加 ONNX Runtime 的库文件路径
target_link_directories(test PRIVATE ${ONNXRUNTIME_DIR}/lib)

# 设置 OpenCV 的路径
set(OpenCV_DIR F:\\cpp_tools\\tools\\opencv_4_5_5_release\\build)

# 查找并添加 OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test PRIVATE ${OpenCV_LIBS})

# 添加 ONNX Runtime 的链接库
target_link_libraries(test PRIVATE onnxruntime)

