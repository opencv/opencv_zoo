#include "opencv2/opencv.hpp"

#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;

std::vector<std::pair<int, int>> backend_target_pairs = {
    {DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
    {DNN_BACKEND_CUDA, DNN_TARGET_CUDA},
    {DNN_BACKEND_CUDA, DNN_TARGET_CUDA_FP16},
    {DNN_BACKEND_TIMVX, DNN_TARGET_NPU},
    {DNN_BACKEND_CANN, DNN_TARGET_NPU}
};

class FER
{
private:
    Net model;
    string modelPath;
    float std[5][2] = {
        {38.2946, 51.6963},
        {73.5318, 51.5014},
        {56.0252, 71.7366},
        {41.5493, 92.3655},
        {70.7299, 92.2041}
    };
    vector<String> expressionEnum = {
        "angry", "disgust", "fearful",
        "happy", "neutral", "sad", "surprised"
    };
    Mat stdPoints = Mat(5, 2, CV_32F, this->std);
    Size patchSize = Size(112,112);
    Scalar imageMean = Scalar(0.5,0.5,0.5);
    Scalar imageStd = Scalar(0.5,0.5,0.5);

    const String inputNames = "data";
    const String outputNames = "label";

    int backend_id;
    int target_id;
    
public:
    FER(const string& modelPath, 
        int backend_id = 0, 
        int target_id = 0) 
      : modelPath(modelPath), backend_id(backend_id), target_id(target_id)
    {
        this->model = readNet(modelPath);
        this->model.setPreferableBackend(backend_id);
        this->model.setPreferableTarget(target_id);
    }

    Mat preprocess(const Mat image, const Mat points)
    {
        // image alignment
        Mat transformation = estimateAffine2D(points, this->stdPoints);
        Mat aligned = Mat::zeros(this->patchSize.height, this->patchSize.width, image.type());    
        warpAffine(image, aligned, transformation, this->patchSize);

        // image normalization
        aligned.convertTo(aligned, CV_32F, 1.0 / 255.0);
        aligned -= imageMean;
        aligned /= imageStd;
        
        return blobFromImage(aligned);;
    }

    String infer(const Mat image, const Mat facePoints)
    {
        Mat points = facePoints(Rect(4, 0, facePoints.cols-5, facePoints.rows)).reshape(2, 5);
        Mat inputBlob = preprocess(image, points);

        this->model.setInput(inputBlob, this->inputNames);
        Mat outputBlob = this->model.forward(this->outputNames);

        Point maxLoc;
        minMaxLoc(outputBlob, nullptr, nullptr, nullptr, &maxLoc);
        
        return getDesc(maxLoc.x);
    }

    String getDesc(int ind) 
    {

        if (ind >= 0 &&  ind < this->expressionEnum.size()) 
        {
            return this->expressionEnum[ind];
        } 
        else 
        {
            cerr << "Error: Index out of bounds." << endl;
            return "";
        }
    }

};

class YuNet
{
public:
    YuNet(const string& model_path,
          const Size& input_size = Size(320, 320),
          float conf_threshold = 0.6f,
          float nms_threshold = 0.3f,
          int top_k = 5000,
          int backend_id = 0,
          int target_id = 0)
        : model_path_(model_path), input_size_(input_size),
          conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
          top_k_(top_k), backend_id_(backend_id), target_id_(target_id)
    {
        model = FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    void setBackendAndTarget(int backend_id, int target_id)
    {
        backend_id_ = backend_id;
        target_id_ = target_id;
        model = FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    /* Overwrite the input size when creating the model. Size format: [Width, Height].
    */
    void setInputSize(const Size& input_size)
    {
        input_size_ = input_size;
        model->setInputSize(input_size_);
    }

    Mat infer(const Mat image)
    {
        Mat res;
        model->detect(image, res);
        return res;
    }

private:
    Ptr<FaceDetectorYN> model;

    string model_path_;
    Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    int backend_id_;
    int target_id_;
};

cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces, const vector<String> expressions, float fps = -1.f)
{
    static cv::Scalar box_color{0, 255, 0};
    static std::vector<cv::Scalar> landmark_color{
        cv::Scalar(255,   0,   0), // right eye
        cv::Scalar(  0,   0, 255), // left eye
        cv::Scalar(  0, 255,   0), // nose tip
        cv::Scalar(255,   0, 255), // right mouth corner
        cv::Scalar(  0, 255, 255)  // left mouth corner
    };
    static cv::Scalar text_color{0, 255, 0};

    auto output_image = image.clone();

    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Expression as text
        String exp = expressions[i];
        cv::putText(output_image, exp, cv::Point(x1, y1+12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);

        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j)
        {
            int x = static_cast<int>(faces.at<float>(i, 2*j+4)), y = static_cast<int>(faces.at<float>(i, 2*j+5));
            cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
        }
    }
    return output_image;
}

string keys =
"{ help  h          |                                                                  | Print help message. }"
"{ model m          | facial_expression_recognition_mobilefacenet_2022july.onnx        | Usage: Path to the model, defaults to facial_expression_recognition_mobilefacenet_2022july.onnx  }"
"{ yunet_model ym   | ../face_detection_yunet/face_detection_yunet_2023mar.onnx        | Usage: Path to the face detection yunet model, defaults to face_detection_yunet_2023mar.onnx  }"
"{ input i          |                                                                  | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ backend_target t | 0                                                                | Choose one of the backend-target pair to run this demo:\n"
                                                                                        "0: (default) OpenCV implementation + CPU,\n"
                                                                                        "1: CUDA + GPU (CUDA),\n"
                                                                                        "2: CUDA + GPU (CUDA FP16),\n"
                                                                                        "3: TIM-VX + NPU,\n"
                                                                                        "4: CANN + NPU}"
"{ save s           | false                                                             | Specify to save results.}"
"{ vis v            | true                                                              | Specify to open a window for result visualization.}"
;


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
 
    parser.about("Facial Expression Recognition");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    string modelPath = parser.get<string>("model");
    string yunetModelPath = parser.get<string>("yunet_model");
    string inputPath = parser.get<string>("input");
    uint8_t backendTarget = parser.get<uint8_t>("backend_target");
    bool saveFlag = parser.get<bool>("save");
    bool visFlag = parser.get<bool>("vis");

    if (modelPath.empty())
        CV_Error(Error::StsError, "Model file " + modelPath + " not found");

    if (yunetModelPath.empty())
        CV_Error(Error::StsError, "Face Detection Model file " + yunetModelPath + " not found");

    YuNet faceDetectionModel(yunetModelPath);
    FER expressionRecognitionModel(modelPath, backend_target_pairs[backendTarget].first, backend_target_pairs[backendTarget].second);

    VideoCapture cap;
    if (!inputPath.empty())
        cap.open(samples::findFile(inputPath));
    else
        cap.open(0);
    
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot opend video or file");    

    Mat frame;
    static const std::string kWinName = "Facial Expression Demo";


    while (waitKey(1) < 0)
    {
        cap >> frame;

        if (frame.empty())
        {
            if(inputPath.empty())
                cout << "Frame is empty" << endl;
            break;
        }

        faceDetectionModel.setInputSize(frame.size());
        
        Mat faces = faceDetectionModel.infer(frame);
        vector<String> expressions;

        for (int i = 0; i < faces.rows; ++i)
        {
            Mat face = faces.row(i);
            String exp = expressionRecognitionModel.infer(frame, face);
            expressions.push_back(exp);

            int x1 = static_cast<int>(faces.at<float>(i, 0));
            int y1 = static_cast<int>(faces.at<float>(i, 1));
            int w = static_cast<int>(faces.at<float>(i, 2));
            int h = static_cast<int>(faces.at<float>(i, 3));
            float conf = faces.at<float>(i, 14);

            std::cout << cv::format("%d: x1=%d, y1=%d, w=%d, h=%d, conf=%.4f expression=%s\n", i, x1, y1, w, h, conf, exp.c_str());

        }

        Mat res_frame = visualize(frame, faces, expressions);

        if(visFlag || inputPath.empty())
        {
            imshow(kWinName, res_frame);
            if(!inputPath.empty())
                waitKey(0);
        }
        if(saveFlag)
        {
            cout << "Results are saved to result.jpg" << endl;

            cv::imwrite("result.jpg", res_frame);
        }
    }
    

    return 0;

}   

