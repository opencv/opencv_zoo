#include <vector>
#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

const auto backendTargetPairs = vector<pair<Backend, Target>>
{
    {DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
    {DNN_BACKEND_CUDA, DNN_TARGET_CUDA},
    {DNN_BACKEND_CUDA, DNN_TARGET_CUDA_FP16},
    {DNN_BACKEND_TIMVX, DNN_TARGET_NPU},
    {DNN_BACKEND_CANN, DNN_TARGET_NPU}
};

const vector<string> nanodetClassLabels = 
{
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
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush" 
};

class NanoDet 
{
public:
    NanoDet(const String& modelPath, const float probThresh = 0.35, const float iouThresh = 0.6, 
            const Backend bId = DNN_BACKEND_DEFAULT, const Target tId = DNN_TARGET_CPU) :
        modelPath(modelPath), probThreshold(probThresh),
        iouThreshold(iouThresh), backendId(bId), targetId(tId), 
        imageShape(416, 416), regMax(7)
    {
        this->strides = { 8, 16, 32, 64 };
        this->net = readNet(modelPath);
        this->net.setPreferableBackend(bId);
        this->net.setPreferableTarget(tId);
        this->project = Mat::zeros(1, this->regMax + 1, CV_32F);
        for (size_t i = 0; i <= this->regMax; ++i) 
        {
            this->project.at<float>(0, i) = static_cast<float>(i);
        }
        this->mean = Scalar(103.53, 116.28, 123.675);
        this->std = Scalar(1.0 / 57.375, 1.0 / 57.12, 1.0 / 58.395);
        this->generateAnchors();
    }

    Mat preProcess(const Mat& inputImage)
    {
        Image2BlobParams paramNanodet;
        paramNanodet.datalayout = DNN_LAYOUT_NCHW;
        paramNanodet.ddepth = CV_32F;
        paramNanodet.mean = this->mean;
        paramNanodet.scalefactor = this->std;
        paramNanodet.size = this->imageShape;
        Mat blob;
        blobFromImageWithParams(inputImage, blob, paramNanodet);
        return blob;
    }

    Mat infer(const Mat& sourceImage)
    {
        Mat blob = this->preProcess(sourceImage);
        this->net.setInput(blob);
        vector<Mat> modelOutput;
        this->net.forward(modelOutput, this->net.getUnconnectedOutLayersNames());
        Mat preds = this->postProcess(modelOutput);
        return preds;
    }

    Mat reshapeIfNeeded(const Mat& input) 
    {
        if (input.dims == 3)
        {
            return input.reshape(0, input.size[1]);
        }
        return input;
    }

    Mat softmaxActivation(const Mat& input) 
    {
        Mat x_exp, x_sum, x_repeat_sum, result;
        exp(input.reshape(0, input.total() / (this->regMax + 1)), x_exp);
        reduce(x_exp, x_sum, 1, REDUCE_SUM, CV_32F);
        repeat(x_sum, 1, this->regMax + 1, x_repeat_sum);
        divide(x_exp, x_repeat_sum, result);
        return result;
    }

    Mat applyProjection(Mat& input) 
    {
        Mat repeat_project;
        repeat(this->project, input.rows, 1, repeat_project);
        multiply(input, repeat_project, input);
        reduce(input, input, 1, REDUCE_SUM, CV_32F);
        Mat projection = input.col(0).clone();
        return projection.reshape(0, projection.total() / 4);
    }

    void preNMS(Mat& anchors, Mat& bbox_pred, Mat& cls_score, const int nms_pre = 1000)
    {
        Mat max_scores;
        reduce(cls_score, max_scores, 1, REDUCE_MAX);

        Mat indices;
        sortIdx(max_scores.t(), indices, SORT_DESCENDING);

        Mat indices_float = indices.colRange(0, nms_pre);
        Mat selected_anchors, selected_bbox_pred, selected_cls_score;
        for (int j = 0; j < indices_float.cols; ++j) 
        {
            selected_anchors.push_back(anchors.row(indices_float.at<int>(j)));
            selected_bbox_pred.push_back(bbox_pred.row(indices_float.at<int>(j)));
            selected_cls_score.push_back(cls_score.row(indices_float.at<int>(j)));
        }

        anchors = selected_anchors;
        bbox_pred = selected_bbox_pred;
        cls_score = selected_cls_score;
    }

    void clipBoundingBoxes(Mat& x1, Mat& y1, Mat& x2, Mat& y2) 
    {
        Mat zeros = Mat::zeros(x1.size(), x1.type());
        x1 = min(max(x1, zeros), Scalar(this->imageShape.width - 1));
        y1 = min(max(y1, zeros), Scalar(this->imageShape.height - 1));
        x2 = min(max(x2, zeros), Scalar(this->imageShape.width - 1));
        y2 = min(max(y2, zeros), Scalar(this->imageShape.height - 1));
    }

    Mat calculateBoundingBoxes(const Mat& anchors, const Mat& bbox_pred) 
    {
        Mat x1 = anchors.col(0) - bbox_pred.col(0);
        Mat y1 = anchors.col(1) - bbox_pred.col(1);
        Mat x2 = anchors.col(0) + bbox_pred.col(2);
        Mat y2 = anchors.col(1) + bbox_pred.col(3);

        clipBoundingBoxes(x1, y1, x2, y2);

        Mat bboxes;
        hconcat(vector<Mat>{x1, y1, x2, y2}, bboxes);

        return bboxes;
    }

    vector<Rect2d> bboxMatToRect2d(const Mat& bboxes)
    {
        Mat bboxes_wh(bboxes.clone());
        bboxes_wh.colRange(2, 4) = bboxes_wh.colRange(2, 4) -= bboxes_wh.colRange(0, 2);
        vector<Rect2d> boxesXYXY;
        for (size_t i = 0; i < bboxes_wh.rows; i++) 
        {
            boxesXYXY.emplace_back(bboxes.at<float>(i, 0),
                                   bboxes.at<float>(i, 1),
                                   bboxes.at<float>(i, 2),
                                   bboxes.at<float>(i, 3));
        }
        return boxesXYXY;
    }

    Mat postProcess(const vector<Mat>& preds)
    {
        vector<Mat> cls_scores, bbox_preds;
        for (size_t i = 0; i < preds.size(); i += 2) 
        {
            cls_scores.push_back(preds[i]);
            bbox_preds.push_back(preds[i + 1]);
        }

        vector<Mat> bboxes_mlvl;
        vector<Mat> scores_mlvl;

        for (size_t i = 0; i < strides.size(); ++i) 
        {
            if (i >= cls_scores.size() || i >= bbox_preds.size()) continue;
            // Extract necessary data
            int stride = strides[i];
            Mat cls_score = reshapeIfNeeded(cls_scores[i]);
            Mat bbox_pred = reshapeIfNeeded(bbox_preds[i]);
            Mat anchors = anchorsMlvl[i].t();

            // Softmax activation, projection, and calculate bounding boxes
            bbox_pred = softmaxActivation(bbox_pred);
            bbox_pred = applyProjection(bbox_pred);
            bbox_pred = stride * bbox_pred;

            const int nms_pre = 1000;
            if (nms_pre > 0 && cls_score.rows > nms_pre) 
            {
                preNMS(anchors, bbox_pred, cls_score, nms_pre);
            }
            
            Mat bboxes = calculateBoundingBoxes(anchors, bbox_pred);

            
            bboxes_mlvl.push_back(bboxes);
            scores_mlvl.push_back(cls_score);
        }
        Mat bboxes;
        Mat scores;
        vconcat(bboxes_mlvl, bboxes);
        vconcat(scores_mlvl, scores);

        vector<Rect2d> boxesXYXY = bboxMatToRect2d(bboxes);
        vector<int> classIds;
        vector<float> confidences;
        for (size_t i = 0; i < scores.rows; ++i) 
        {
            Point maxLoc;
            minMaxLoc(scores.row(i), nullptr, nullptr, nullptr, &maxLoc);
            classIds.push_back(maxLoc.x);
            confidences.push_back(scores.at<float>(i, maxLoc.x));
        }

        vector<int> indices;
        NMSBoxesBatched(boxesXYXY, confidences, classIds, probThreshold, iouThreshold, indices);
        Mat result(int(indices.size()), 6, CV_32FC1);
        int row = 0;
        for (auto idx : indices)
        {
            bboxes.rowRange(idx, idx + 1).copyTo(result(Rect(0, row, 4, 1)));
            result.at<float>(row, 4) = confidences[idx];
            result.at<float>(row, 5) = static_cast<float>(classIds[idx]);
            row++;
        }
        if (indices.size() == 0)
        {
            return Mat();
        }
        return result;
    }

    void generateAnchors()
    {
        for (const int stride : strides) {
            int feat_h = this->imageShape.height / stride;
            int feat_w = this->imageShape.width / stride;
            
            vector<Mat> anchors;
            
            for (int y = 0; y < feat_h; ++y) 
            {
                for (int x = 0; x < feat_w; ++x) 
                {
                    float shift_x = x * stride;
                    float shift_y = y * stride;
                    float cx = shift_x + 0.5 * (stride - 1);
                    float cy = shift_y + 0.5 * (stride - 1);
                    Mat anchor_point = (Mat_<float>(2, 1) << cx, cy);
                    anchors.push_back(anchor_point);
                }
            }
            Mat anchors_mat;
            hconcat(anchors, anchors_mat);
            this->anchorsMlvl.push_back(anchors_mat);
        }
    }
private:
    Net net;
    String modelPath;
    vector<int> strides;
    Size imageShape;
    int regMax;
    float probThreshold;
    float iouThreshold;
    Backend backendId;
    Target targetId;
    Mat project;
    Scalar mean;
    Scalar std;
    vector<Mat> anchorsMlvl;
};

// Function to resize and pad an image and return both the image and scale information
tuple<Mat, vector<double>> letterbox(const Mat& sourceImage, const Size& target_size = Size(416, 416)) 
{
    Mat img = sourceImage.clone();

    double top = 0, left = 0, newh = target_size.height, neww = target_size.width;

    if (img.rows != img.cols) 
    {
        double hw_scale = static_cast<double>(img.rows) / img.cols;
        if (hw_scale > 1) 
        {
            newh = target_size.height;
            neww = static_cast<int>(target_size.width / hw_scale);
            resize(img, img, Size(neww, newh), 0, 0, INTER_AREA);
            left = static_cast<int>((target_size.width - neww) * 0.5);
            copyMakeBorder(img, img, 0, 0, left, target_size.width - neww - left, BORDER_CONSTANT, Scalar(0));
        } 
        else 
        {
            newh = static_cast<int>(target_size.height * hw_scale);
            neww = target_size.width;
            resize(img, img, Size(neww, newh), 0, 0, INTER_AREA);
            top = static_cast<int>((target_size.height - newh) * 0.5);
            copyMakeBorder(img, img, top, target_size.height - newh - top, 0, 0, BORDER_CONSTANT, Scalar(0));
        }
    } 
    else 
    {
        resize(img, img, target_size, 0, 0, INTER_AREA);
    }
    vector<double> letterbox_scale = {top, left, newh, neww};

    return make_tuple(img, letterbox_scale);
}

// Function to scale bounding boxes back to original image coordinates
vector<int> unletterbox(const Mat& bbox, const Size& original_image_shape, const vector<double>& letterbox_scale) 
{
    vector<int> ret(bbox.cols);

    int h = original_image_shape.height;
    int w = original_image_shape.width;
    double top = letterbox_scale[0];
    double left = letterbox_scale[1];
    double newh = letterbox_scale[2];
    double neww = letterbox_scale[3];

    if (h == w) 
    {
        double ratio = static_cast<double>(h) / newh;
        for (int& val : ret) 
        {
            val = static_cast<int>(val * ratio);
        }
        return ret;
    }

    double ratioh = static_cast<double>(h) / newh;
    double ratiow = static_cast<double>(w) / neww;
    ret[0] = max(static_cast<int>((bbox.at<float>(0) - left) * ratiow), 0);
    ret[1] = max(static_cast<int>((bbox.at<float>(1) - top) * ratioh), 0);
    ret[2] = min(static_cast<int>((bbox.at<float>(2) - left) * ratiow), w);
    ret[3] = min(static_cast<int>((bbox.at<float>(3) - top) * ratioh), h);

    return ret;
}

// Function to visualize predictions on an image
Mat visualize(const Mat& preds, const Mat& result_image, const vector<double>& letterbox_scale, bool video, double fps = 0.0) 
{
    Mat visualized_image = result_image.clone();

    // Draw FPS if provided
    if (fps > 0.0 && video) 
    {
        std::ostringstream fps_stream;
        fps_stream << "FPS: " << std::fixed << std::setprecision(2) << fps;
        putText(visualized_image, fps_stream.str(), Point(10, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
    }

    // Draw bounding boxes and labels for each prediction
    for (size_t i = 0; i < preds.rows; i++) 
    {
        Mat pred = preds.row(i);
        Mat bbox = pred.colRange(0, 4);
        double conf = pred.at<float>(4);
        int classid = static_cast<int>(pred.at<float>(5));

        // Convert bbox coordinates back to original image space
        vector<int> unnormalized_bbox = unletterbox(bbox, visualized_image.size(), letterbox_scale);

        // Draw bounding box
        rectangle(visualized_image, Point(unnormalized_bbox[0], unnormalized_bbox[1]),
                      Point(unnormalized_bbox[2], unnormalized_bbox[3]), Scalar(0, 255, 0), 2);

        // Draw label
        stringstream label;
        label << nanodetClassLabels[classid] << ": " << fixed << setprecision(2) << conf;
        putText(visualized_image, label.str(), Point(unnormalized_bbox[0], unnormalized_bbox[1] - 10),
                  FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    }

    return visualized_image;
}

void processImage(Mat& inputImage, NanoDet& nanodet, TickMeter& tm, bool save, bool vis, bool video)
{
    cvtColor(inputImage, inputImage, COLOR_BGR2RGB);
    tuple<Mat, vector<double>> w = letterbox(inputImage);
    Mat inputBlob = get<0>(w);
    vector<double> letterboxScale  = get<1>(w);
    
    tm.start();
    Mat predictions = nanodet.infer(inputBlob);
    tm.stop();
    if (!video)
    {
        cout << "Inference time: " << tm.getTimeMilli() << " ms\n";
    }

    Mat img = visualize(predictions, inputImage, letterboxScale, video, tm.getFPS());
    cvtColor(img, img, COLOR_BGR2RGB);
    if (save)
    {
        static const string kOutputName = "result.jpg";
        imwrite(kOutputName, img);
        if (!video)
        {
            cout << "Results saved to " + kOutputName << endl;
        }
    }
    if (vis)
    {
        static const string kWinName = "model";
        imshow(kWinName, img);
    }
}


const String keys =
        "{ help  h          |                                               | Print help message. }"
        "{ model m          | object_detection_nanodet_2022nov.onnx         | Usage: Path to the model, defaults to object_detection_nanodet_2022nov.onnx  }"
        "{ input i          |                                               | Path to the input image. Omit for using the default camera.}"
        "{ confidence       | 0.35                                          | Class confidence }"
        "{ nms              | 0.6                                           | Enter nms IOU threshold }"
        "{ save s           | true                                          | Specify to save results. This flag is invalid when using the camera. }"
        "{ vis v            | true                                          | Specify to open a window for result visualization. This flag is invalid when using the camera. }"
        "{ backend bt       | 0                                             | Choose one of computation backends: "
        "0: (default) OpenCV implementation + CPU, "
        "1: CUDA + GPU (CUDA), "
        "2: CUDA + GPU (CUDA FP16), "
        "3: TIM-VX + NPU, "
        "4: CANN + NPU}";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser.about("Use this script to run Nanodet inference using OpenCV, a contribution by Sri Siddarth Chakaravarthy as part of GSOC_2022.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string model = parser.get<String>("model");
    string inputPath = parser.get<String>("input");
    float confThreshold = parser.get<float>("confidence");
    float nmsThreshold = parser.get<float>("nms");
    bool save = parser.get<bool>("save");
    bool vis = parser.get<bool>("vis");
    int backendTargetid = parser.get<int>("backend");

    if (model.empty())
    {
        CV_Error(Error::StsError, "Model file " + model + " not found");
    }

    NanoDet nanodet(model, confThreshold, nmsThreshold,
                    backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);

    TickMeter tm;
    if (parser.has("input"))
    {
        Mat inputImage = imread(samples::findFile(inputPath));
        static const bool kNotVideo = false;
        processImage(inputImage, nanodet, tm, save, vis, kNotVideo);
        waitKey(0);
    }
    else
    {
        VideoCapture cap;
        cap.open(0);
        if (!cap.isOpened())
        {
            CV_Error(Error::StsError, "Cannot open video or file");
        }

        Mat frame;
        while (waitKey(1) < 0)
        {
            cap >> frame;
            if (frame.empty())
            {
                cout << "Frame is empty" << endl;
                waitKey();
                break;
            }
            tm.reset();
            static const bool kIsVideo = true;
            processImage(frame, nanodet, tm, save, vis, kIsVideo);
        }
        cap.release();
    }
    return 0;
}
