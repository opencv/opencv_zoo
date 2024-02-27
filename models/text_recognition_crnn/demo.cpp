#include <iostream>
#include <codecvt>


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "charset_32_94_3944.h"

using namespace std;
using namespace cv;
using namespace dnn;

vector< pair<cv::dnn::Backend, cv::dnn::Target> > backendTargetPairs = {
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU),
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA),
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA_FP16),
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_TIMVX, dnn::DNN_TARGET_NPU),
        std::make_pair<cv::dnn::Backend, cv::dnn::Target>(dnn::DNN_BACKEND_CANN, dnn::DNN_TARGET_NPU)};
        
vector<u16string> loadCharset(string);

std::string keys =
"{ help  h           |                                              | Print help message. }"
"{ model m           | text_recognition_CRNN_EN_2021sep.onnx        | Usage: Set model type, defaults to text_recognition_CRNN_EN_2021sep.onnx }"
"{ input i           |                                              | Usage: Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ width             | 736                                          | Usage: Resize input image to certain width, default = 736. It should be multiple by 32.}"
"{ height            | 736                                          | Usage: Resize input image to certain height, default = 736. It should be multiple by 32.}"
"{ binary_threshold  | 0.3                                          | Usage: Threshold of the binary map, default = 0.3.}"
"{ polygon_threshold | 0.5                                          | Usage: Threshold of polygons, default = 0.5.}"
"{ max_candidates    | 200                                          | Usage: Set maximum number of polygon candidates, default = 200.}"
"{ unclip_ratio      | 2.0                                          | Usage: The unclip ratio of the detected text region, which determines the output size, default = 2.0.}"
"{ save s            | 1                                            | Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.}"
"{ viz v             | 1                                            | Usage: Specify to open a new window to show results.}"
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



class CRNN {
private:
    string modelPath;
    dnn::Backend backendId;
    dnn::Target targetId;
    Net model;
    vector<u16string> charset;
    Size inputSize;
    Mat targetVertices;

public:
    CRNN(string modPath, dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU) : modelPath(modPath), backendId(bId), targetId(tId) {

        this->model = readNet(this->modelPath);
        this->model.setPreferableBackend(this->backendId);
        this->model.setPreferableTarget(this->targetId);
        // load charset by the name of model
        if (this->modelPath.find("_EN_") != string::npos)
            this->charset = loadCharset("CHARSET_EN_36");
        else if (this->modelPath.find("_CH_") != string::npos)
            this->charset = loadCharset("CHARSET_CH_94");
        else if (this->modelPath.find("_CN_") != string::npos)
            this->charset = loadCharset("CHARSET_CN_3944");
        else
            CV_Error(-1, "Charset not supported! Exiting ...");

        this->inputSize = Size(100, 32); // Fixed
        this->targetVertices = Mat(4, 1, CV_32FC2);
        this->targetVertices.row(0) = Vec2f(0, this->inputSize.height - 1);
        this->targetVertices.row(1) = Vec2f(0, 0);
        this->targetVertices.row(2) = Vec2f(this->inputSize.width - 1, 0);
        this->targetVertices.row(3) = Vec2f(this->inputSize.width - 1, this->inputSize.height - 1);
    }

    Mat preprocess(Mat image, Mat rbbox)
	{
        // Remove conf, reshape and ensure all is np.float32
        Mat vertices;
        rbbox.reshape(2, 4).convertTo(vertices, CV_32FC2);

        Mat rotationMatrix = getPerspectiveTransform(vertices, this->targetVertices);
        Mat cropped;
        warpPerspective(image, cropped, rotationMatrix, this->inputSize);

        // 'CN' can detect digits (0\~9), upper/lower-case letters (a\~z and A\~Z), and some special characters
        // 'CH' can detect digits (0\~9), upper/lower-case le6tters (a\~z and A\~Z), some Chinese characters and some special characters
        if (this->modelPath.find("CN") == string::npos && this->modelPath.find("CH") == string::npos)
            cvtColor(cropped, cropped, COLOR_BGR2GRAY);
        Mat blob = blobFromImage(cropped, 1 / 127.5, this->inputSize, Scalar::all(127.5));
        return blob;
    }

    u16string infer(Mat image, Mat rbbox)
	{
        // Preprocess
        Mat inputBlob = this->preprocess(image, rbbox);

        //  Forward
        this->model.setInput(inputBlob);
        Mat outputBlob = this->model.forward();

        // Postprocess
        u16string results = this->postprocess(outputBlob);

        return results;
    }

    u16string postprocess(Mat outputBlob)
	{
        // Decode charaters from outputBlob
        Mat character = outputBlob.reshape(1, outputBlob.size[0]);
        u16string    text(u"");
        for (int i = 0; i < character.rows; i++)
        {
            double minVal, maxVal;
            Point maxIdx;
            minMaxLoc(character.row(i), &minVal, &maxVal, nullptr, &maxIdx);
            if (maxIdx.x != 0)
                text += charset[maxIdx.x - 1];
            else
                text += u"-";
        }
        // adjacent same letters as well as background text must be removed to get the final output
        u16string    textFilter(u"");

        for (int i = 0; i < text.size(); i++)
            if (text[i] != u'-' && !(i > 0 && text[i] == text[i - 1]))
                textFilter += text[i];
        return textFilter;
    }
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

    parser.about("An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (https://arxiv.org/abs/1507.05717)");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int backendTargetid = parser.get<int>("backend");
    String modelPath = parser.get<String>("model");

    if (modelPath.empty())
    {
        CV_Error(Error::StsError, "Model file " + modelPath + " not found");
    }

    Size inpSize(parser.get<int>("width"), parser.get<int>("height"));
    float binThresh = parser.get<float>("binary_threshold");
    float polyThresh = parser.get<float>("polygon_threshold");
    int maxCand = parser.get<int>("max_candidates");
    double unRatio = parser.get<float>("unclip_ratio");
    bool save = parser.get<bool>("save");
    bool viz = parser.get<float>("viz");

    PPOCRDet detector("../text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx", inpSize, binThresh, polyThresh, maxCand, unRatio, backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);
    CRNN recognizer(modelPath, backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);
    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        cap.open(0);
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot open video or file");
    Mat originalImage;
    static const std::string kWinName = modelPath;
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

        // inference of text detector
        TickMeter tm;
        tm.start();
        pair< vector<vector<Point>>, vector<float> > results = detector.infer(image);
        tm.stop();
        if (results.first.size() > 0 && results.second.size() > 0)
        {
            u16string texts;
            auto score=results.second.begin();
            for (auto box : results.first)
            {
                Mat result = Mat(box).reshape(2, 4);
                texts = texts + u"'" + recognizer.infer(image, result) + u"'";
            }
            std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> converter;
            std::cout << converter.to_bytes(texts) << std::endl;
        }
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
                imshow(kWinName, originalImage);
        }
        else
            imshow(kWinName, originalImage);
  
    }
    return 0;
}
