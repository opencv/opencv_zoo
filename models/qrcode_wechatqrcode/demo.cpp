#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/wechat_qrcode.hpp>
#include <string>
#include <vector>

class WeChatQRCode {
   public:
    WeChatQRCode(const std::string& detect_prototxt,
                 const std::string& detect_model,
                 const std::string& sr_prototxt, const std::string& sr_model,
                 int backend_target_index)
        : backend_target_index_(backend_target_index) {
        
        const std::vector<std::pair<int, int>> backend_target_pairs = {
            {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
            {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
            {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16},
            {cv::dnn::DNN_BACKEND_TIMVX, cv::dnn::DNN_TARGET_NPU},
            {cv::dnn::DNN_BACKEND_CANN, cv::dnn::DNN_TARGET_NPU}};

        if (backend_target_index_ < 0 ||
            backend_target_index_ >= backend_target_pairs.size()) {
            throw std::invalid_argument("Invalid backend-target index");
        }

        // initialize detector
        detector_ = cv::makePtr<cv::wechat_qrcode::WeChatQRCode>(
            detect_prototxt, detect_model, sr_prototxt, sr_model);
    }

    std::pair<std::vector<std::string>, std::vector<cv::Mat>> detect(
        const cv::Mat& image) {
        std::vector<std::string> results;
        std::vector<cv::Mat> points;
        results = detector_->detectAndDecode(image, points);
        return {results, points};
    }

    cv::Mat visualize(const cv::Mat& image,
                      const std::vector<std::string>& results,
                      const std::vector<cv::Mat>& points,
                      cv::Scalar points_color = cv::Scalar(0, 255, 0),
                      cv::Scalar text_color = cv::Scalar(0, 255, 0),
                      double fps = -1) const {
        cv::Mat output = image.clone();

        if (fps >= 0) {
            cv::putText(output, "FPS: " + std::to_string(fps), cv::Point(0, 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color);
        }

        double fontScale = 0.5;
        int fontSize = 1;

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& p = points[i];

            for (int r = 0; r < p.rows; ++r) {
                cv::Point point(p.at<float>(r, 0), p.at<float>(r, 1));
                cv::circle(output, point, 10, points_color, -1);
            }

            int qrcode_center_x = (p.at<float>(0, 0) + p.at<float>(2, 0)) / 2;
            int qrcode_center_y = (p.at<float>(0, 1) + p.at<float>(2, 1)) / 2;

            int baseline = 0;
            cv::Size text_size =
                cv::getTextSize(results[i], cv::FONT_HERSHEY_DUPLEX, fontScale,
                                fontSize, &baseline);

            cv::Point text_pos(qrcode_center_x - text_size.width / 2,
                               qrcode_center_y + text_size.height / 2);

            cv::putText(output, results[i], text_pos, cv::FONT_HERSHEY_DUPLEX,
                        fontScale, text_color, fontSize);
        }

        return output;
    }

   private:
    int backend_target_index_;
    cv::Ptr<cv::wechat_qrcode::WeChatQRCode> detector_;
};

int main(int argc, char** argv) {
    
    cv::CommandLineParser parser(
        argc, argv,
        "{help h                |                             | Show this help message.}"
        "{input i               |                             | Set path to the input image. Omit for using default camera.}"
        "{detect_prototxt_path  | detect_2021nov.prototxt     | Set path to detect.prototxt.}"
        "{detect_model_path     | detect_2021nov.caffemodel   | Set path to detect.caffemodel.}"
        "{sr_prototxt_path      | sr_2021nov.prototxt         | Set path to sr.prototxt.}"
        "{sr_model_path         | sr_2021nov.caffemodel       | Set path to sr.caffemodel.}"
        "{backend_target bt     | 0                           | Choose one of the backend-target pairs to run this demo.}"
        "{save s                | false                       | Specify to save file with results.}"
        "{vis v                 | false                       | Specify to open a new window to show results.}");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // get paths
    std::string detect_prototxt = parser.get<std::string>("detect_prototxt_path");
    std::string detect_model = parser.get<std::string>("detect_model_path");
    std::string sr_prototxt = parser.get<std::string>("sr_prototxt_path");
    std::string sr_model = parser.get<std::string>("sr_model_path");
    int backend_target_index = parser.get<int>("backend_target");

    // input check
    std::string input_path = parser.get<std::string>("input");
    bool save_result = parser.get<bool>("save");
    bool visualize_result = parser.get<bool>("vis");

    try {
        WeChatQRCode qrDetector(detect_prototxt, detect_model, sr_prototxt,
                                sr_model, backend_target_index);

        if (!input_path.empty()) {
            // process image
            cv::Mat image = cv::imread(input_path);
            if (image.empty()) {
                std::cerr << "Could not read the image" << std::endl;
                return -1;
            }

            std::pair<std::vector<std::string>, std::vector<cv::Mat>> detectionResult = qrDetector.detect(image);
            auto& results = detectionResult.first;
            auto& points = detectionResult.second;

            for (const auto& result : results) {
                std::cout << result << std::endl;
            }

            cv::Mat result_image = qrDetector.visualize(image, results, points);

            if (save_result) {
                cv::imwrite("result.jpg", result_image);
                std::cout << "Results saved to result.jpg" << std::endl;
            }

            if (visualize_result) {
                cv::imshow(input_path, result_image);
                cv::waitKey(0);
            }
        } else {
            // process camera
            cv::VideoCapture cap(0);
            if (!cap.isOpened()) {
                std::cerr << "Error opening camera" << std::endl;
                return -1;
            }

            cv::Mat frame;
            cv::TickMeter tm;

            while (true) {
                cap >> frame;
                if (frame.empty()) {
                    std::cout << "No frames grabbed" << std::endl;
                    break;
                }

                std::pair<std::vector<std::string>, std::vector<cv::Mat>> detectionResult = qrDetector.detect(frame);
                auto& results = detectionResult.first;
                auto& points = detectionResult.second;

                tm.start();
                double fps = tm.getFPS();
                tm.stop();

                cv::Mat result_frame = qrDetector.visualize(
                    frame, results, points, cv::Scalar(0, 255, 0),
                    cv::Scalar(0, 255, 0), fps);
                cv::imshow("WeChatQRCode Demo", result_frame);

                tm.reset();

                if (cv::waitKey(1) >= 0) break;
            }
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
