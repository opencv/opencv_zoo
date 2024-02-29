#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

vector< pair<cv::dnn::Backend, cv::dnn::Target> > backendTargetPairs = {
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU),
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA),
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA_FP16),
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_TIMVX, dnn::DNN_TARGET_NPU),
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_CANN, dnn::DNN_TARGET_NPU)};
        

std::string keys =
"{ help  h           |                                              | Print help message. }"
"{ model m           | text_detection_cn_ppocrv3_2023may.onnx       | Usage: Set model type, defaults to text_detection_ch_ppocrv3_2023may.onnx }"
"{ input i           |                                              | Usage: Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ width             | 736                                          | Usage: Resize input image to certain width, default = 736. It should be multiple by 32.}"
"{ height            | 736                                          | Usage: Resize input image to certain height, default = 736. It should be multiple by 32.}"
"{ binary_threshold  | 0.3                                          | Usage: Threshold of the binary map, default = 0.3.}"
"{ polygon_threshold | 0.5                                          | Usage: Threshold of polygons, default = 0.5.}"
"{ max_candidates    | 200                                          | Usage: Set maximum number of polygon candidates, default = 200.}"
"{ unclip_ratio      | 2.0                                          | Usage: The unclip ratio of the detected text region, which determines the output size, default = 2.0.}"
"{ save s            | true                                         | Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.}"
"{ viz v             | true                                         | Usage: Specify to open a new window to show results. Invalid in case of camera input.}"
"{ backend bt        | 0                                            | Choose one of computation backends: "
"0: (default) OpenCV implementation + CPU, "
"1: CUDA + GPU (CUDA), "
"2: CUDA + GPU (CUDA FP16), "
"3: TIM-VX + NPU, "
"4: CANN + NPU}";


class PPOCRDet {
public:

    PPOCRDet(string modPath, Size inSize = Size(736, 736), float binThresh = 0.3,
        float polyThresh = 0.5, int maxCand = 200, double unRatio = 2.0,
        dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU) : modelPath(modPath), inputSize(inSize), binaryThreshold(binThresh),
        polygonThreshold(polyThresh), maxCandidates(maxCand), unclipRatio(unRatio),
        backendId(bId), targetId(tId)
    {
        this->model = TextDetectionModel_DB(readNet(modelPath));
        this->model.setPreferableBackend(backendId);
        this->model.setPreferableTarget(targetId);

        this->model.setBinaryThreshold(binaryThreshold);
        this->model.setPolygonThreshold(polygonThreshold);
        this->model.setUnclipRatio(unclipRatio);
        this->model.setMaxCandidates(maxCandidates);

        this->model.setInputParams(1.0 / 255.0, inputSize, Scalar(122.67891434, 116.66876762, 104.00698793));
    }
    pair< vector<vector<Point>>, vector<float> > infer(Mat image) {
        CV_Assert(image.rows == this->inputSize.height && "height of input image != net input size ");
        CV_Assert(image.cols == this->inputSize.width && "width of input image != net input size ");
        vector<vector<Point>> pt;
        vector<float> confidence;
        this->model.detect(image, pt, confidence);
        return make_pair< vector<vector<Point>> &, vector< float > &>(pt, confidence);
    }

private:
    string modelPath;
    TextDetectionModel_DB model;
    Size inputSize;
    float binaryThreshold;
    float polygonThreshold;
    int maxCandidates;
    double unclipRatio;
    dnn::Backend backendId;
    dnn::Target targetId;

};

Mat visualize(Mat image, pair< vector<vector<Point>>, vector<float> >&results, double fps=-1, Scalar boxColor=Scalar(0, 255, 0), Scalar textColor=Scalar(0, 0, 255), bool isClosed=true, int thickness=2)
{
    Mat output;
    image.copyTo(output);
    if (fps > 0)
        putText(output, format("FPS: %.2f", fps), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, textColor);
    polylines(output, results.first, isClosed, boxColor, thickness);
    return output;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser.about("Use this program to run Real-time Scene Text Detection with Differentiable Binarization in opencv Zoo  using OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int backendTargetid = parser.get<int>("backend");
    String modelName = parser.get<String>("model");

    if (modelName.empty())
    {
        CV_Error(Error::StsError, "Model file " + modelName + " not found");
    }

    Size inpSize(parser.get<int>("width"), parser.get<int>("height"));
    float binThresh = parser.get<float>("binary_threshold");
    float polyThresh = parser.get<float>("polygon_threshold");
    int maxCand = parser.get<int>("max_candidates");
    double unRatio = parser.get<float>("unclip_ratio");
    bool save = parser.get<bool>("save");
    bool viz = parser.get<bool>("viz");

    PPOCRDet model(modelName, inpSize, binThresh, polyThresh, maxCand, unRatio, backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);

    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(0);
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot open video or file");
    Mat originalImage;
    static const std::string kWinName = modelName;
    while (waitKey(1) < 0)
    {
        cap >> originalImage;
        if (originalImage.empty())
        {
            if (parser.has("input"))
            {
                cout << "Frame is empty" << endl;
                break;
            }
            else
                continue;
        }
        int originalW = originalImage.cols;
        int originalH = originalImage.rows;
        double scaleHeight = originalH / double(inpSize.height);
        double scaleWidth = originalW / double(inpSize.width);
        Mat image;
        resize(originalImage, image, inpSize);

        // inference
        TickMeter tm;
        tm.start();
        pair< vector<vector<Point>>, vector<float> > results = model.infer(image);
        tm.stop();
        auto x = results.first;
        // Scale the results bounding box
        for (auto &pts : results.first)
        {
            for (int i = 0; i < 4; i++)
            {
                pts[i].x = int(pts[i].x * scaleWidth);
                pts[i].y = int(pts[i].y * scaleHeight);
            }
        }
        originalImage = visualize(originalImage, results, tm.getFPS());
        tm.reset();
        if (parser.has("input"))
        {
            if (save)
            {
                cout << "Result image saved to result.jpg\n";
                imwrite("result.jpg", originalImage);
            }
            if (viz)
            {
                imshow(kWinName, originalImage);
                waitKey(0);
            }
        }
        else
            imshow(kWinName, originalImage);
    }
    return 0;
}


