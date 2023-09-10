#include <windows.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <locale>
#include <codecvt>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/imgproc/types_c.h>

// 命名空间
using namespace std;

typedef int (*DetectFunction)(const char*, const char*);

// 定义BoxInfo结构类型
typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

int main()
{
    //--------------测试使用的jpg图片数据--------------
    cv::Mat image = cv::imread("1.jpg");
    // 获取图像的宽度和高度
    uint width = image.cols;
    uint height = image.rows;
    cv::Mat yuvImage;
    cv::cvtColor(image, yuvImage, cv::COLOR_BGR2YUV_I420);
    //--------------测试使用的jpg图片数据--------------
    // 获取YUV420图像数据指针
    const char* image_buff = reinterpret_cast<const char*>(yuvImage.data);
    string classes[7] = {"crack", "nick", "bend", "bend", "corrosion", "coat_shedding", "dent"};

    // 加载DLL
    HMODULE dllHandle = LoadLibrary("new_inference.dll");

    if (dllHandle != NULL) {
        // 获取导出函数地址
        auto detectFunction = reinterpret_cast<int(*)(const char*, uint, uint, std::vector<BoxInfo>&)>(GetProcAddress(dllHandle, "detect_per_YUV420_image"));

        if (detectFunction != nullptr) {
            std::vector<BoxInfo> result;

            // 调用导出函数
            int returnValue = detectFunction(image_buff, width, height, result);

            // 在result中获取检测结果
            std::cout << "outer result.size(): " << result.size() << std::endl;

            // //循环
            for (size_t i = 0; i < result.size(); i++)
            {
            	int x1 = int(result[i].x1);
            	int y1 = int(result[i].y1);
                int x2 = int(result[i].x2);
                int y2 = int(result[i].y2);
                std::cout << "x1:" << x1 << ", y1:" << y1 << "; x2:" << x2 << ", y2:" << std::endl;
            	// rectangle(frame, Point(xmin, ymin), Point(int(result[i].x2), int(result[i].y2)), Scalar(0, 0, 255), 2);
            	float score = result[i].score;
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string formattedScore = ss.str();
            	string label = classes[result[i].label] + ":" + formattedScore;
                std::cout << "label: " << label << std::endl;
            	//putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
            }

        } else {

            std::cerr << "Failed to get function address." << std::endl;
        }

        // 卸载DLL
        FreeLibrary(dllHandle);
    } else {
        std::cerr << "Failed to load DLL." << std::endl;
    }

    return 0;
}




