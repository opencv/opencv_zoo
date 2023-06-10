#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace dnn;

std::string keys =
"{ help  h          | | Print help message. }"
"{ model m             | image_classification_mobilenetv1_2022apr.onnx | An optional path to file with preprocessing parameters }"
"{ classes             | labels_imagenet_1k.txt |path to labels file  }"
"{ input i          | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ initial_width    | 0 | Preprocess input image by initial resizing to a specific width.}"
"{ initial_height   | 0 | Preprocess input image by initial resizing to a specific height.}"
"{ std              | 1.0 1.0 1.0 | Preprocess input image by dividing on a standard deviation.}"
"{ rgb              | true | swap R and B plane.}"
"{ crop             | false | Preprocess input image by center cropping.}"
"{ backend          | 0 | Choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation, "
"4: VKCOM, "
"5: CUDA, "
"6: WebNN }"
"{ target           | 0 | Choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU, "
"4: Vulkan, "
"6: CUDA, "
"7: CUDA fp16 (half-float preprocess) }";

using namespace std;
using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

int main(int argc, char** argv)
{
    std::vector<std::string> classes;
    CommandLineParser parser(argc, argv, keys);

    parser.about("Use this script to run classification deep learning networks in opencv Zoo  using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    Scalar mean(123.675, 116.28, 103.53);
    Scalar std(255*0.229, 255*0.224, 255*0.225);
    double scale = 1.0;
    Size inpSize(224, 224);
    int rszWidth = parser.get<int>("initial_width");
    int rszHeight = parser.get<int>("initial_height");
    bool swapRB = parser.get<bool>("rgb");
    bool crop = parser.get<bool>("crop");
    String model = samples::findFile(parser.get<String>("model"));
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");

    if (model.empty())
    {
        CV_Error(Error::StsError, "Model file " + model + " not found");
    }
    if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }
    Net net = readNet(model);
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(0, CAP_DSHOW);
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot opend video or file");
    Mat frame, blob;
    static const std::string kWinName = model;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Frame is empty" << endl;
            waitKey();
            break;
        }

        if (rszWidth != 0 && rszHeight != 0)
        {
            resize(frame, frame, Size(rszWidth, rszHeight));
        }

        //! [Create a 4D blob from a frame]
        blobFromImage(frame, blob, scale, inpSize, mean, swapRB, crop);

        // Check std values.
        if (std.val[0] != 0.0 && std.val[1] != 0.0 && std.val[2] != 0.0)
        {
            // Divide blob by std.
            divide(blob, std, blob);
        }
        //! [Set input blob]
        net.setInput(blob);
        int classId;
        double confidence;
        Mat prob = net.forward();

        //! [Get a class with a highest score]
        Point classIdPoint;
        minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        classId = classIdPoint.x;
        std::string label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
            classes[classId].c_str()),
            confidence);
        putText(frame, label, Point(0, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        imshow(kWinName, frame);
    }
    return 0;
}
