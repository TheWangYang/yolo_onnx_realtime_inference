#include <fstream>
#include <sstream>
#include <iostream>
#include <locale>
#include <codecvt>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/imgproc/types_c.h>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

// 命名空间
using namespace std;
using namespace cv;
using namespace Ort;

// 自定义配置结构
struct Configuration
{
	public: 
	float confThreshold; // Confidence threshold置信度阈值
	float nmsThreshold;  // Non-maximum suppression threshold非最大抑制阈值
	float objThreshold;  //Object Confidence threshold对象置信度阈值
	string modelpath;
};

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

class YOLOv5
{
public:
	YOLOv5(Configuration config);
	void detect(Mat& frame);
private:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	int num_classes;
	string classes[7] = {"crack", "nick", "bend", "bend", "corrosion", "coat_shedding", "dent"};

	const bool keep_ratio = true;
	vector<float> input_image_;		// 输入图片
	void normalize_(Mat img);		// 归一化函数
	void nms(vector<BoxInfo>& input_boxes);  
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5s"); // 初始化环境
	Session *ort_session = nullptr;    // 初始化Session指针选项
	SessionOptions sessionOptions = SessionOptions();  //初始化Session对象
	//SessionOptions sessionOptions;
	vector<char*> input_names;  // 定义一个字符指针vector
	vector<char*> output_names; // 定义一个字符指针vector
	vector<vector<int64_t>> input_node_dims; // >=1 outputs  ，二维vector
	vector<vector<int64_t>> output_node_dims; // >=1 outputs ,int64_t C/C++标准
};

YOLOv5::YOLOv5(Configuration config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	this->num_classes = 7;//sizeof(this->classes)/sizeof(this->classes[0]);  // 类别数量
	this->inpHeight = 640;
	this->inpWidth = 640;
	
	//string model_path = config.modelpath;
    wstring_convert<codecvt_utf8<wchar_t>> converter;
    wstring model_path = converter.from_bytes(config.modelpath);

    //OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

	// 设置图优化类型
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();  //输入输出节点数量                         
	size_t numOutputNodes = ort_session->GetOutputCount(); 
	AllocatorWithDefaultOptions allocator;   // 配置输入输出节点内存
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));		// 内存
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);   // 类型
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();  // 
		auto input_dims = input_tensor_info.GetShape();    // 输入shape
		input_node_dims.push_back(input_dims);	// 保存
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];      // 5+classes
	this->num_proposal = output_node_dims[0][1];  // pre_box

}


Mat YOLOv5::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)  //修改图片大小并填充边界防止失真
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);  //等比例缩小，防止失真
			*top = (int)(this->inpHeight - *newh) * 0.5;  //上部缺失部分
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114); //上部填补top大小，下部填补剩余部分，左右不填补
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLOv5::normalize_(Mat img)  //归一化
{
	//    img.convertTo(img, CV_32F);
    //cout<<"picture size"<<img.rows<<img.cols<<img.channels()<<endl;
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());  // vector大小

	for (int c = 0; c < 3; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // 行
		{
			for (int j = 0; j < col; j++)  // 列
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];  // Mat里的ptr函数访问任意一行像素的首地址,2-c:表示rgb
				this->input_image_[c * row * col + i * col + j] = pix / 255.0; //将每个像素块归一化后装进容器
			}
		}
	}
}

void YOLOv5::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列
	vector<bool> remove_flags(input_boxes.size(),false);
	auto iou = [](const BoxInfo& box1,const BoxInfo& box2)
	{
		float xx1 = max(box1.x1, box2.x1);
		float yy1 = max(box1.y1, box2.y1);
		float xx2 = min(box1.x2, box2.x2);
		float yy2 = min(box1.y2, box2.y2);
		// 交集
		float w = max(0.0f, xx2 - xx1 + 1);
		float h = max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// 并集
		float union_area = max(0.0f,box1.x2-box1.x1) * max(0.0f,box1.y2-box1.y1)
						   + max(0.0f,box2.x2-box2.x1) * max(0.0f,box2.y2-box2.y1) - inter_area;
		return inter_area / union_area;
	};
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if(remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if(remove_flags[j]) continue;
			if(input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i],input_boxes[j])>=this->nmsThreshold)
			{
				remove_flags[j] = true;
			}
		}
	}
	int idx_t = 0;
    // remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}

void YOLOv5::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);   //改大小后做padding防失真
	this->normalize_(dstimg);       //归一化
	// 定义一个输入矩阵，int64_t是下面作为输入参数时的类型
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };  //1,3,640,640

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    //使用Ort库创建一个输入张量，其中包含了需要进行目标检测的图像数据。
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	//generate proposals
    //cout<<"ort_outputs_size"<<ort_outputs.size()<<endl;
	vector<BoxInfo> generate_boxes;  // BoxInfo自定义的结构体
    float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;  //原图高和新高比，原图宽与新宽比

	float* pdata = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData

	for(int i = 0; i < num_proposal; ++i) // 遍历所有的num_pre_boxes
	{   
		int index = i * nout;      // prob[b*num_pred_boxes*(classes+5)]  
		float obj_conf = pdata[index + 4];  // 置信度分数
        //cout<<"k"<<obj_conf<<endl;
		if (obj_conf > this->objThreshold)  // 大于阈值
		{
			int class_idx = 0;
			float max_class_socre = 0;
			for (int k = 0; k < this->num_classes; ++k)
			{
				if (pdata[k + index + 5] > max_class_socre)
				{
					max_class_socre = pdata[k + index + 5];
					class_idx = k;
				}
			}
			max_class_socre *= obj_conf;   // 最大的类别分数*置信度
			if (max_class_socre > this->confThreshold) // 再次筛选
			{ 
				//const int class_idx = classIdPoint.x;
				float cx = pdata[index];  //x
				float cy = pdata[index+1];  //y
				float w = pdata[index+2];  //w
				float h = pdata[index+3];  //h
				//cout<<cx<<cy<<w<<h<<endl;
				float xmin = (cx - padw - 0.5 * w)*ratiow;
				float ymin = (cy - padh - 0.5 * h)*ratioh;
				float xmax = (cx - padw + 0.5 * w)*ratiow;
				float ymax = (cy - padh + 0.5 * h)*ratioh;
				//cout<<xmin<<ymin<<xmax<<ymax<<endl;
				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx });
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->classes[generate_boxes[i].label] + ":" + label;
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}


int main(int argc, char* argv[])
{

    //计算FPS
    double totalTime = 0.0;
    int frameCount = 0;
    double fps = 0.0;

	int camera_index = 0;  // 默认摄像头索引为 0
    if (argc > 1)
    {
        camera_index = std::atoi(argv[1]);  // 获取命令行参数中的摄像头索引
    }
    cv::VideoCapture capture(camera_index);  // 使用默认的摄像头设备

	if (!capture.isOpened())
	{
		cout << "Failed to open the video device." << endl;
		return -1;
	}

	Configuration yolo_nets = { 0.3, 0.5, 0.3, "./weights/best.onnx" }; // 初始化属性
	YOLOv5 yolo_model(yolo_nets);

	clock_t startTime, endTime; // 计算时间

	Mat frame;
	while (true)
	{
		capture.read(frame); // 读取视频帧
		if (frame.empty())
		{
			cout << "Failed to capture a frame." << endl;
			break;
		}

		double timeStart = (double)getTickCount();
		startTime = clock(); // 计时开始

        // 计算FPS
        double start = cv::getTickCount();
		yolo_model.detect(frame);
        double end = cv::getTickCount();
        double elapsed = (end - start) / cv::getTickFrequency();
		endTime = clock(); // 计时结束
		double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
		cout << "clock_running time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
		cout << "The run time is: " << (double)clock() / CLOCKS_PER_SEC << "s" << endl;
		cout << "getTickCount_running time: " << nTime << " sec" << endl;

        totalTime += elapsed;
        frameCount++;

        fps = frameCount / totalTime;
        std::stringstream ss;
        ss << "FPS: " << fps;
        cv::putText(frame, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        totalTime = 0.0;
        frameCount = 0;

		cv::imshow("Object Detection", frame);

		if (cv::waitKey(1) == 'q')
		{
			cout << "----------------------------------" << endl;
			cout << "------------- closed -------------" << endl;
			cout << "----------------------------------" << endl;
			break;
		}
	}

	capture.release();
	cv::destroyAllWindows();

	return 0;
}
