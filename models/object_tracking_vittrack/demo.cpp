#include <iostream>
#include <opencv2/opencv.hpp>

struct TrackingResult {
    bool isLocated;
    cv::Rect bbox;
    float score;
};

class VitTrack {
public:

    VitTrack(const std::string& model_path, int backend_id = 0, int target_id = 0) {
        params.net = model_path;
        params.backend = backend_id;
        params.target = target_id;
        model = cv::TrackerVit::create(params);
    }

    void init(const cv::Mat& image, const cv::Rect& roi) {
        model->init(image, roi);
    }

    TrackingResult infer(const cv::Mat& image) {
        TrackingResult result;
        result.isLocated = model->update(image, result.bbox);
        result.score = model->getTrackingScore();
        return result;
    }

private:
    cv::TrackerVit::Params params;
    cv::Ptr<cv::TrackerVit> model;
};

cv::Mat visualize(const cv::Mat& image, const cv::Rect& bbox, float score, bool isLocated, double fps = -1.0,
                  const cv::Scalar& box_color = cv::Scalar(0, 255, 0), const cv::Scalar& text_color = cv::Scalar(0, 255, 0),
                  double fontScale = 1.0, int fontSize = 1) {
    cv::Mat output = image.clone();
    int h = output.rows;
    int w = output.cols;

    if (fps >= 0) {
        cv::putText(output, "FPS: " + std::to_string(fps), cv::Point(0, 30), cv::FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize);
    }

    if (isLocated && score >= 0.3) {
        cv::rectangle(output, bbox, box_color, 2);
        cv::putText(output, cv::format("%.2f", score), cv::Point(bbox.x, bbox.y + 25),
                    cv::FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize);
    } else {
        cv::Size text_size = cv::getTextSize("Target lost!", cv::FONT_HERSHEY_DUPLEX, fontScale, fontSize, nullptr);
        int text_x = (w - text_size.width) / 2;
        int text_y = (h - text_size.height) / 2;
        cv::putText(output, "Target lost!", cv::Point(text_x, text_y), cv::FONT_HERSHEY_DUPLEX, fontScale, cv::Scalar(0, 0, 255), fontSize);
    }

    return output;
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv,
        "{input i           |                                       |Set path to the input video. Omit for using default camera.}"
        "{model_path        |object_tracking_vittrack_2023sep.onnx  |Set model path}"
        "{backend_target bt |0                                      |Choose backend-target pair: 0 - OpenCV implementation + CPU, 1 - CUDA + GPU (CUDA), 2 - CUDA + GPU (CUDA FP16), 3 - TIM-VX + NPU, 4 - CANN + NPU}");

    std::string input_path = parser.get<std::string>("input");
    std::string model_path = parser.get<std::string>("model_path");
    int backend_target = parser.get<int>("backend_target");

    std::vector<std::vector<int>> backend_target_pairs = {
        {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16},
        {cv::dnn::DNN_BACKEND_TIMVX, cv::dnn::DNN_TARGET_NPU},
        {cv::dnn::DNN_BACKEND_CANN, cv::dnn::DNN_TARGET_NPU}
    };

    int backend_id = backend_target_pairs[backend_target][0];
    int target_id = backend_target_pairs[backend_target][1];

    // Create VitTrack tracker
    VitTrack tracker(model_path, backend_id, target_id);

    // Open video capture
    cv::VideoCapture video;
    if (input_path.empty()) {
        video.open(0);  // Default camera
    } else {
        video.open(input_path);
    }

    if (!video.isOpened()) {
        std::cerr << "Error: Could not open video source" << std::endl;
        return -1;
    }

    // Select an object
    cv::Mat first_frame;
    video >> first_frame;

    if (first_frame.empty()) {
        std::cerr << "No frames grabbed!" << std::endl;
        return -1;
    }

    cv::Mat first_frame_copy = first_frame.clone();
    cv::putText(first_frame_copy, "1. Drag a bounding box to track.", cv::Point(0, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::putText(first_frame_copy, "2. Press ENTER to confirm", cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::Rect roi = cv::selectROI("VitTrack Demo", first_frame_copy);

    if (roi.area() == 0) {
        std::cerr << "No ROI is selected! Exiting..." << std::endl;
        return -1;
    } else {
        std::cout << "Selected ROI: " << roi << std::endl;
    }

    // Initialize tracker with ROI
    tracker.init(first_frame, roi);

    // Track frame by frame
    cv::TickMeter tm;
    while (cv::waitKey(1) < 0) {
        video >> first_frame;
        if (first_frame.empty()) {
            std::cout << "End of video" << std::endl;
            break;
        }

        // Inference
        tm.start();
        TrackingResult result = tracker.infer(first_frame);
        tm.stop();

        // Visualize
        cv::Mat frame = first_frame.clone();
        frame = visualize(frame, result.bbox, result.score, result.isLocated, tm.getFPS());
        cv::imshow("VitTrack Demo", frame);
        tm.reset();
    }

    return 0;
}
