#include <assert.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <onnxruntime_c_api.h>
#include <opencv2/opencv.hpp>

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

//*****************************************************************************
// helper function to check for status
void CheckStatus(OrtStatus* status)
{
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    // initialize environment...one environment per process
    // environment maintains thread pools and other state info
    OrtEnv* env;
    CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

    // initialize session options if needed
    OrtSessionOptions* session_options;
    CheckStatus(g_ort->CreateSessionOptions(&session_options));
    g_ort->SetIntraOpNumThreads(session_options, 1);
    g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

    // create session and load model into memory
    OrtSession* session;
    const wchar_t* model_path = L"./weights/best.onnx";
    CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session));

    // print model input layer information
    size_t num_input_nodes;
    OrtStatus* status;
    OrtAllocator* allocator;
    CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));
    status = g_ort->SessionGetInputCount(session, &num_input_nodes);
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;

    printf("Number of inputs = %zu\n", num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) {
        char* input_name;
        status = g_ort->SessionGetInputName(session, i, allocator, &input_name);
        printf("Input %zu : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        OrtTypeInfo* typeinfo;
        status = g_ort->SessionGetInputTypeInfo(session, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
        printf("Input %zu : type=%d\n", i, type);

        size_t num_dims;
        CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
        printf("Input %zu : num_dims=%zu\n", i, num_dims);
        input_node_dims.resize(num_dims);
        g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
        for (size_t j = 0; j < num_dims; j++)
            printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);

        g_ort->ReleaseTypeInfo(typeinfo);
    }

    // print model output layer information
    size_t num_output_nodes;
    status = g_ort->SessionGetOutputCount(session, &num_output_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);

    printf("Number of outputs = %zu\n", num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; i++) {
        char* output_name;
        status = g_ort->SessionGetOutputName(session, i, allocator, &output_name);
        printf("Output %zu : name=%s\n", i, output_name);
        output_node_names[i] = output_name;

        OrtTypeInfo* typeinfo;
        status = g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
        printf("Output %zu : type=%d\n", i, type);

        size_t num_dims;
        CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
        printf("Output %zu : num_dims=%zu\n", i, num_dims);
        std::vector<int64_t> output_node_dims(num_dims);
        g_ort->GetDimensions(tensor_info, output_node_dims.data(), num_dims);
        for (size_t j = 0; j < num_dims; j++)
            printf("Output %zu : dim %zu=%jd\n", i, j, output_node_dims[j]);

        g_ort->ReleaseTypeInfo(typeinfo);
    }

    // open video capture
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        printf("Failed to open video capture.\n");
        return 1;
    }

    cv::Mat frame;
    cv::namedWindow("Camera", cv::WINDOW_NORMAL);

    // process frames from the video capture
    while (true) {
        // read frame from video capture
        capture >> frame;
        if (frame.empty())
            break;

        // preprocess frame if needed
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 640));  // adjust size as per model input requirements
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);  // adjust color format as per model input requirements

        // prepare input tensor
        size_t input_tensor_size = resized.rows * resized.cols * resized.channels();
        std::vector<float> input_tensor_values(input_tensor_size);
        for (int row = 0; row < resized.rows; row++) {
            for (int col = 0; col < resized.cols; col++) {
                cv::Vec3b pixel = resized.at<cv::Vec3b>(row, col);
                for (int channel = 0; channel < resized.channels(); channel++) {
                    input_tensor_values[row * resized.cols * resized.channels() + col * resized.channels() + channel] = pixel[channel] / 255.0;
                }
            }
        }

        // create input tensor
        OrtMemoryInfo* memory_info;
        CheckStatus(g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info));
        OrtValue* input_tensor = NULL;
        g_ort->CreateTensorAsOrtValue(allocator, input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
        CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), input_node_dims.data(), input_node_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
        int is_tensor;
        CheckStatus(g_ort->IsTensor(input_tensor, &is_tensor));
        assert(is_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);

        // run inference
        OrtValue* output_tensor = NULL;
        CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), num_output_nodes, &output_tensor));
        CheckStatus(g_ort->IsTensor(output_tensor, &is_tensor));
        assert(is_tensor);


        // get output tensor data
        float* floatarr;
        CheckStatus(g_ort->GetTensorMutableData(output_tensor, (void**)&floatarr));

        for (int i = 0; i < 7; i++){
            printf("Score for class [%d] =  %f\n", i, floatarr[i]);
        }

        // release input and output tensors
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseValue(output_tensor);

        // display the frame
        cv::imshow("Camera", frame);

        // check for 'q' key to exit
        if (cv::waitKey(1) == 'q')
            break;
    }

    // release video capture and OpenCV windows
    capture.release();
    cv::destroyAllWindows();

    // clean up
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    printf("Done!\n");
    return 0;
}
