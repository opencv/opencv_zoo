#include <vector>
#include <string>
#include <utility>

#include <opencv2/opencv.hpp>
#include "labelsimagenet1k.h"

using namespace std;
using namespace cv;
using namespace dnn;

vector< pair<dnn::Backend, dnn::Target> > backendTargetPairs = {
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA_FP16),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_TIMVX, dnn::DNN_TARGET_NPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CANN, dnn::DNN_TARGET_NPU) };


std::string keys =
"{ help  h          |                                               | Print help message. }"
"{ model m          | image_classification_mobilenetv1_2022apr.onnx | Usage: Set model type, defaults to image_classification_mobilenetv1_2022apr.onnx (v1) }"
"{ input i          |                                               | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ initial_width    | 0                                             | Preprocess input image by initial resizing to a specific width.}"
"{ initial_height   | 0                                             | Preprocess input image by initial resizing to a specific height.}"
"{ rgb              | true                                          | swap R and B plane.}"
"{ crop             | false                                         | Preprocess input image by center cropping.}"
"{ vis v            | true                                          | Usage: Specify to open a new window to show results.}"
"{ backend bt       | 0                                             | Choose one of computation backends: "
"0: (default) OpenCV implementation + CPU, "
"1: CUDA + GPU (CUDA), "
"2: CUDA + GPU (CUDA FP16), "
"3: TIM-VX + NPU, "
"4: CANN + NPU}";


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser.about("Use this script to run classification deep learning networks in opencv Zoo using OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int rszWidth = parser.get<int>("initial_width");
    int rszHeight = parser.get<int>("initial_height");
    bool swapRB = parser.get<bool>("rgb");
    bool crop = parser.get<bool>("crop");
    bool vis = parser.get<bool>("vis");
    String model = parser.get<String>("model");
    int backendTargetid = parser.get<int>("backend");

    if (model.empty())
    {
        CV_Error(Error::StsError, "Model file " + model + " not found");
    }
    vector<string> labels = getLabelsImagenet1k();

    Net net = readNet(samples::findFile(model));
    net.setPreferableBackend(backendTargetPairs[backendTargetid].first);
    net.setPreferableTarget(backendTargetPairs[backendTargetid].second);
    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(samples::findFile(parser.get<String>("input")));
    else
        cap.open(0);
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot open video or file");
    Mat frame, blob;
    static const std::string kWinName = model;
    int nbInference = 0;
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
        Image2BlobParams paramMobilenet;
        paramMobilenet.datalayout = DNN_LAYOUT_NCHW;
        paramMobilenet.ddepth = CV_32F;
        paramMobilenet.mean = Scalar(123.675, 116.28, 103.53);
        paramMobilenet.scalefactor = Scalar(1 / (255. * 0.229), 1 / (255. * 0.224), 1 / (255. * 0.225));
        paramMobilenet.size = Size(224, 224);
        paramMobilenet.swapRB = swapRB;
        if (crop)
            paramMobilenet.paddingmode = DNN_PMODE_CROP_CENTER;
        else
            paramMobilenet.paddingmode = DNN_PMODE_NULL;
        //! [Create a 4D blob from a frame]
        blobFromImageWithParams(frame, blob, paramMobilenet);

        //! [Set input blob]
        net.setInput(blob);
        Mat prob = net.forward();

        //! [Get a class with a highest score]
        Point classIdPoint;
        double confidence;
        minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;
        std::string label = format("%s: %.4f", (labels.empty() ? format("Class #%d", classId).c_str() :
            labels[classId].c_str()),
            confidence);
        if (vis)
        {
            putText(frame, label, Point(0, 55), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
            imshow(kWinName, frame);
        }
        else
        {
            cout << label << endl;
            nbInference++;
            if (nbInference > 100)
            {
                cout << nbInference << " inference made. Demo existing" << endl;
                break;
            }
        }
    }
    return 0;
}
