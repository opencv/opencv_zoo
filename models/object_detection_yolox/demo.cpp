#include <vector>
#include <string>
#include <utility>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

vector< pair<dnn::Backend, dnn::Target> > backendTargetPairs = {
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA_FP16),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_TIMVX, dnn::DNN_TARGET_NPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CANN, dnn::DNN_TARGET_NPU) };

vector<string> labelYolox = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

class YoloX {
private:
    Net net;
    string modelPath;
    Size inputSize;
    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    dnn::Backend backendId;
    dnn::Target targetId;
    int num_classes;
    vector<int> strides;
    Mat expandedStrides;
    Mat grids;

public:
    YoloX(string modPath, float confThresh = 0.35, float nmsThresh = 0.5, float objThresh = 0.5, dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU) :
        modelPath(modPath), confThreshold(confThresh),
        nmsThreshold(nmsThresh), objThreshold(objThresh),
        backendId(bId), targetId(tId)
    {
        this->num_classes = int(labelYolox.size());
        this->net = readNet(modelPath);
        this->inputSize = Size(640, 640);
        this->strides = vector<int>{ 8, 16, 32 };
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
        this->generateAnchors();
    }

    Mat preprocess(Mat img)
    {
        Mat blob;
        Image2BlobParams paramYolox;
        paramYolox.datalayout = DNN_LAYOUT_NCHW;
        paramYolox.ddepth = CV_32F;
        paramYolox.mean = Scalar::all(0);
        paramYolox.scalefactor = Scalar::all(1);
        paramYolox.size = Size(img.cols, img.rows);
        paramYolox.swapRB = true;

        blob = blobFromImageWithParams(img, paramYolox);
        return blob;
    }

   Mat infer(Mat srcimg)
    {
        Mat inputBlob = this->preprocess(srcimg);

        this->net.setInput(inputBlob);
        vector<Mat> outs;
        this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

        Mat predictions = this->postprocess(outs[0]);
        return predictions;
    }

    Mat postprocess(Mat outputs)
    {
        Mat dets = outputs.reshape(0,outputs.size[1]);
        Mat col01;
        add(dets.colRange(0, 2), this->grids, col01);
        Mat col23;
        exp(dets.colRange(2, 4), col23);
        vector<Mat> col = { col01, col23 };
        Mat boxes;
        hconcat(col, boxes);
        float* ptr = this->expandedStrides.ptr<float>(0);
        for (int r = 0; r < boxes.rows; r++, ptr++)
        {
            boxes.rowRange(r, r + 1) = *ptr * boxes.rowRange(r, r + 1);
        }
        // get boxes
        Mat boxes_xyxy(boxes.rows, boxes.cols, CV_32FC1, Scalar(1));
        Mat scores = dets.colRange(5, dets.cols).clone();
        vector<float> maxScores(dets.rows);
        vector<int> maxScoreIdx(dets.rows);
        vector<Rect2d> boxesXYXY(dets.rows);

        for (int r = 0; r < boxes_xyxy.rows; r++, ptr++)
        {
            boxes_xyxy.at<float>(r, 0) = boxes.at<float>(r, 0) - boxes.at<float>(r, 2) / 2.f;
            boxes_xyxy.at<float>(r, 1) = boxes.at<float>(r, 1) - boxes.at<float>(r, 3) / 2.f;
            boxes_xyxy.at<float>(r, 2) = boxes.at<float>(r, 0) + boxes.at<float>(r, 2) / 2.f;
            boxes_xyxy.at<float>(r, 3) = boxes.at<float>(r, 1) + boxes.at<float>(r, 3) / 2.f;
            // get scores and class indices
            scores.rowRange(r, r + 1) = scores.rowRange(r, r + 1) * dets.at<float>(r, 4);
            double minVal, maxVal;
            Point maxIdx;
            minMaxLoc(scores.rowRange(r, r+1), &minVal, &maxVal, nullptr, &maxIdx);
            maxScoreIdx[r] = maxIdx.x;
            maxScores[r] = float(maxVal);
            boxesXYXY[r].x = boxes_xyxy.at<float>(r, 0);
            boxesXYXY[r].y = boxes_xyxy.at<float>(r, 1);
            boxesXYXY[r].width = boxes_xyxy.at<float>(r, 2);
            boxesXYXY[r].height = boxes_xyxy.at<float>(r, 3);
        }

        vector<int> keep;
        NMSBoxesBatched(boxesXYXY, maxScores, maxScoreIdx, this->confThreshold, this->nmsThreshold, keep);
        Mat candidates(int(keep.size()), 6, CV_32FC1);
        int row = 0;
        for (auto idx : keep)
        {
            boxes_xyxy.rowRange(idx, idx + 1).copyTo(candidates(Rect(0, row, 4, 1)));
            candidates.at<float>(row, 4) = maxScores[idx];
            candidates.at<float>(row, 5) = float(maxScoreIdx[idx]);
            row++;
        }
        if (keep.size() == 0)
            return Mat();
        return candidates;
        
    }


    void generateAnchors()
    {
        vector< tuple<int, int, int> > nb;
        int total = 0;

        for (auto v : this->strides)
        {
            int w = this->inputSize.width / v;
            int h = this->inputSize.height / v;
            nb.push_back(tuple<int, int, int>(w * h, w, v));
            total += w * h;
        }
        this->grids = Mat(total, 2, CV_32FC1);
        this->expandedStrides = Mat(total, 1, CV_32FC1);
        float* ptrGrids = this->grids.ptr<float>(0);
        float* ptrStrides = this->expandedStrides.ptr<float>(0);
        int pos = 0;
        for (auto le : nb)
        {
            int r = get<1>(le);
            for (int i = 0; i < get<0>(le); i++, pos++)
            {
                *ptrGrids++ = float(i % r);
                *ptrGrids++ = float(i / r);
                *ptrStrides++ = float((get<2>(le)));
            }
        }
    }
};

std::string keys =
"{ help  h          |                                               | Print help message. }"
"{ model m          | object_detection_yolox_2022nov.onnx           | Usage: Path to the model, defaults to object_detection_yolox_2022nov.onnx  }"
"{ input i          |                                               | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ confidence       | 0.5                                           | Class confidence }"
"{ obj              | 0.5                                           | Enter object threshold }"
"{ nms              | 0.5                                           | Enter nms IOU threshold }"
"{ save s           | true                                          | Specify to save results. This flag is invalid when using camera. }"
"{ vis v            | 1                                             | Specify to open a window for result visualization. This flag is invalid when using camera. }"
"{ backend bt       | 0                                             | Choose one of computation backends: "
"0: (default) OpenCV implementation + CPU, "
"1: CUDA + GPU (CUDA), "
"2: CUDA + GPU (CUDA FP16), "
"3: TIM-VX + NPU, "
"4: CANN + NPU}";

pair<Mat, double> letterBox(Mat srcimg, Size targetSize = Size(640, 640))
{
    Mat paddedImg(targetSize.height, targetSize.width, CV_32FC3, Scalar::all(114.0));
    Mat resizeImg;

    double ratio = min(targetSize.height / double(srcimg.rows), targetSize.width / double(srcimg.cols));
    resize(srcimg, resizeImg, Size(int(srcimg.cols * ratio), int(srcimg.rows * ratio)), INTER_LINEAR);
    resizeImg.copyTo(paddedImg(Rect(0, 0, int(srcimg.cols * ratio), int(srcimg.rows * ratio))));
    return pair<Mat, double>(paddedImg, ratio);
}

Mat unLetterBox(Mat bbox, double letterboxScale)
{
    return bbox / letterboxScale;
}

Mat visualize(Mat dets, Mat srcimg, double letterbox_scale, double fps = -1)
{
    Mat resImg = srcimg.clone();

    if (fps > 0)
        putText(resImg, format("FPS: %.2f", fps), Size(10, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

    for (int row = 0; row < dets.rows; row++)
    {
        Mat boxF = unLetterBox(dets(Rect(0, row, 4, 1)), letterbox_scale);
        Mat box;
        boxF.convertTo(box, CV_32S);
        float score = dets.at<float>(row, 4);
        int clsId = int(dets.at<float>(row, 5));

        int x0 = box.at<int>(0, 0);
        int y0 = box.at<int>(0, 1);
        int x1 = box.at<int>(0, 2);
        int y1 = box.at<int>(0, 3);

        string text = format("%s : %f", labelYolox[clsId].c_str(), score * 100);
        int font = FONT_HERSHEY_SIMPLEX;
        int baseLine = 0;
        Size txtSize = getTextSize(text, font, 0.4, 1, &baseLine);
        rectangle(resImg, Point(x0, y0), Point(x1, y1), Scalar(0, 255, 0), 2);
        rectangle(resImg, Point(x0, y0 + 1), Point(x0 + txtSize.width + 1, y0 + int(1.5 * txtSize.height)), Scalar(255, 255, 255), -1);
        putText(resImg, text, Point(x0, y0 + txtSize.height), font, 0.4, Scalar(0, 0, 0), 1);
    }

    return resImg;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser.about("Use this script to run Yolox deep learning networks in opencv_zoo using OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string model = parser.get<String>("model");
    float confThreshold = parser.get<float>("confidence");
    float objThreshold = parser.get<float>("obj");
    float nmsThreshold = parser.get<float>("nms");
    bool vis = parser.get<bool>("vis");
    bool save = parser.get<bool>("save");
    int backendTargetid = parser.get<int>("backend");

    if (model.empty())
    {
        CV_Error(Error::StsError, "Model file " + model + " not found");
    }

    YoloX modelNet(model, confThreshold, nmsThreshold, objThreshold,
        backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);
    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(samples::findFile(parser.get<String>("input")));
    else
        cap.open(0);
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot open video or file");
    Mat frame, inputBlob;
    double letterboxScale;

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
        pair<Mat, double> w = letterBox(frame);
        inputBlob = get<0>(w);
        letterboxScale  = get<1>(w);
        TickMeter tm;
        tm.start();
        Mat predictions = modelNet.infer(inputBlob);
        tm.stop();
        cout << "Inference time: " << tm.getTimeMilli() << " ms\n";
        Mat img = visualize(predictions, frame, letterboxScale, tm.getFPS());
        if (save && parser.has("input"))
        {
            imwrite("result.jpg", img);
        }
        if (vis)
        {
            imshow(kWinName, img);
        }
    }
    return 0;
}
