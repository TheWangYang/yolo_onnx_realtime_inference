#include <windows.h>

typedef int (*DetectFunction)(const char*, const char*);

int main()
{
    HINSTANCE hDLL = LoadLibrary("new_inference.dll");
    if (hDLL == NULL)
    {
        // Handle DLL loading error
        return 1;
    }

    DetectFunction detectFunc = (DetectFunction)GetProcAddress(hDLL, "detect_per_image");
    if (detectFunc == NULL)
    {
        // Handle function not found error
        return 2;
    }

    const char* inputVideoPath = "test.jpg";
    const char* outputVideoPath = "result.jpg";

    int result = detectFunc(inputVideoPath, outputVideoPath);

    FreeLibrary(hDLL);

    return result;
}




