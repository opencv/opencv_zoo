#include <iostream>
#include <opencv2/opencv.hpp>

class VitTrack {
public:

    VitTrack(const std::string& model_path, int backend_id = 0, int target_id = 0)
        : model_path(model_path), backend_id(backend_id), target_id(target_id) {
        params.net = model_path;
        params.backend = backend_id;
        params.target = target_id;

        model = cv::TrackerVit::create(params);
    }

    ~VitTrack() = default;

    const std::string& getName() const {
        static std::string name = "VitTrack";
        return name;
    }

    void setBackendAndTarget(int backend_id, int target_id) {
        this->backend_id = backend_id;
        this->target_id = target_id;

        params.backend = backend_id;
        params.target = target_id;

        model = cv::TrackerVit::create(params);
        if (!model) {
            std::cerr << "Error: Failed to create the VIT tracker" << std::endl;
        }
    }

    void init(const cv::Mat& image, const cv::Rect& roi) {
        if (model) {
            model->init(image, roi);
        } else {
            std::cerr << "Error: VIT tracker not initialized" << std::endl;
        }
    }

    std::tuple<bool, cv::Rect, float> infer(const cv::Mat& image) {
        bool is_located = false;
        cv::Rect bbox;
        float score = 0.0;

        if (model) {
            is_located = model->update(image, bbox);
            score = model->getTrackingScore();
        } else {
            std::cerr << "Error: VIT tracker not initialized" << std::endl;
        }

        return std::make_tuple(is_located, bbox, score);
    }

private:
    std::string model_path;
    int backend_id;
    int target_id;
    cv::TrackerVit::Params params;
    cv::Ptr<cv::TrackerVit> model;
};

#include <iostream>
#include <opencv2/opencv.hpp>

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
        "{input i| |Set path to the input video. Omit for using default camera.}"
        "{model_path |object_tracking_vittrack_2023sep.onnx|Set model path}"
        "{backend_target bt|0|Choose backend-target pair: 0 - OpenCV implementation + CPU, 1 - CUDA + GPU (CUDA), 2 - CUDA + GPU (CUDA FP16), 3 - TIM-VX + NPU, 4 - CANN + NPU}"
        "{save s|false|Specify to save a file with results. Invalid in case of camera input.}"
        "{vis v|false|Specify to open a new window to show results. Invalid in case of camera input.}");

    std::string input_path = parser.get<std::string>("input");
    std::string model_path = parser.get<std::string>("model_path");
    int backend_target = parser.get<int>("backend_target");
    bool save_results = parser.get<bool>("save");
    bool visualize_results = parser.get<bool>("vis");

    // Check OpenCV version
    if (CV_VERSION_MAJOR < 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR < 9)) {
        std::cerr << "Please install the latest opencv version (>=4.9.0)" << std::endl;
        return -1;
    }

    // Valid combinations of backends and targets
    std::vector<std::vector<int>> backend_target_pairs = {
        {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16},
        {cv::dnn::DNN_BACKEND_TIMVX, cv::dnn::DNN_TARGET_NPU},
        {cv::dnn::DNN_BACKEND_CANN, cv::dnn::DNN_TARGET_NPU}
    };

    int backend_id = backend_target_pairs[backend_target][0];
    int target_id = backend_target_pairs[backend_target][1];

    // Create VitTrack model
    VitTrack model(model_path, backend_id, target_id);

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
    cv::putText(first_frame_copy, "1. Drag a bounding box to track.", cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::putText(first_frame_copy, "2. Press ENTER to confirm", cv::Point(0, 35), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::Rect roi = cv::selectROI("vitTrack Demo", first_frame_copy);

    if (roi.area() == 0) {
        std::cerr << "No ROI is selected! Exiting..." << std::endl;
        return -1;
    } else {
        std::cout << "Selected ROI: " << roi << std::endl;
    }

    // Initialize tracker with ROI
    model.init(first_frame, roi);

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
        bool isLocated;
        cv::Rect bbox;
        float score;
        std::tie(isLocated, bbox, score) = model.infer(first_frame);
        tm.stop();

        // Visualize
        cv::Mat frame = first_frame.clone();
        frame = visualize(frame, bbox, score, isLocated, tm.getFPS());
        cv::imshow("VitTrack Demo", frame);
        tm.reset();
    }

    return 0;
}
