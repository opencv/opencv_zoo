#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

const std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>>
    backend_target_pairs = {
        {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16},
        {cv::dnn::DNN_BACKEND_TIMVX, cv::dnn::DNN_TARGET_NPU},
        {cv::dnn::DNN_BACKEND_CANN, cv::dnn::DNN_TARGET_NPU}};

struct Detection {
    std::array<cv::Point2f, 4> corners;
    float score;
};

class LPDYuNet {
   public:
    LPDYuNet(const std::string& model_path,
             const cv::Size& input_size = cv::Size(320, 240),
             float conf_threshold = 0.8f, float nms_threshold = 0.3f,
             int top_k = 5000, int keep_top_k = 750,
             int backend_id = cv::dnn::DNN_BACKEND_DEFAULT,
             int target_id = cv::dnn::DNN_TARGET_CPU)
        : model_path_(model_path),
          input_size_(input_size),
          conf_threshold_(conf_threshold),
          nms_threshold_(nms_threshold),
          top_k_(top_k),
          keep_top_k_(keep_top_k),
          backend_id_(backend_id),
          target_id_(target_id) {
        model_ = cv::dnn::readNet(model_path_);
        model_.setPreferableBackend(backend_id_);
        model_.setPreferableTarget(target_id_);
        generatePriors();
    }

    void setInputSize(const cv::Size& input_size) {
        if (input_size == input_size_) {
            return;
        }
        input_size_ = input_size;
        generatePriors();
    }

    std::vector<Detection> infer(const cv::Mat& image) {
        if (image.size() != input_size_) {
            throw std::runtime_error(
                cv::format("Input image size (%d x %d) does not match model "
                           "input size (%d x %d)",
                           image.cols, image.rows, input_size_.width,
                           input_size_.height));
        }

        cv::Mat input_blob = cv::dnn::blobFromImage(image);
        model_.setInput(input_blob);

        std::vector<cv::Mat> output_blobs;
        model_.forward(output_blobs, output_names_);

        return postprocess(output_blobs);
    }

   private:
    void generatePriors() {
        const int w = input_size_.width;
        const int h = input_size_.height;

        const int feature_map_2th_h = ((h + 1) / 2) / 2;
        const int feature_map_2th_w = ((w + 1) / 2) / 2;
        const int feature_map_3th_h = feature_map_2th_h / 2;
        const int feature_map_3th_w = feature_map_2th_w / 2;
        const int feature_map_4th_h = feature_map_3th_h / 2;
        const int feature_map_4th_w = feature_map_3th_w / 2;
        const int feature_map_5th_h = feature_map_4th_h / 2;
        const int feature_map_5th_w = feature_map_4th_w / 2;
        const int feature_map_6th_h = feature_map_5th_h / 2;
        const int feature_map_6th_w = feature_map_5th_w / 2;

        const std::array<std::pair<int, int>, 4> feature_maps = {
            std::make_pair(feature_map_3th_h, feature_map_3th_w),
            std::make_pair(feature_map_4th_h, feature_map_4th_w),
            std::make_pair(feature_map_5th_h, feature_map_5th_w),
            std::make_pair(feature_map_6th_h, feature_map_6th_w)};

        priors_.clear();
        for (size_t k = 0; k < feature_maps.size(); ++k) {
            const int feature_h = feature_maps[k].first;
            const int feature_w = feature_maps[k].second;
            for (int i = 0; i < feature_h; ++i) {
                for (int j = 0; j < feature_w; ++j) {
                    for (const int min_size : min_sizes_[k]) {
                        const float s_kx =
                            static_cast<float>(min_size) / static_cast<float>(w);
                        const float s_ky =
                            static_cast<float>(min_size) / static_cast<float>(h);
                        const float cx =
                            (static_cast<float>(j) + 0.5f) *
                            static_cast<float>(steps_[k]) / static_cast<float>(w);
                        const float cy =
                            (static_cast<float>(i) + 0.5f) *
                            static_cast<float>(steps_[k]) / static_cast<float>(h);
                        priors_.push_back({cx, cy, s_kx, s_ky});
                    }
                }
            }
        }
    }

    std::vector<Detection> postprocess(const std::vector<cv::Mat>& blob) const {
        if (blob.size() != 3) {
            throw std::runtime_error(
                cv::format("Expected 3 output blobs, got %zu", blob.size()));
        }

        cv::Mat loc =
            blob[0].reshape(1, static_cast<int>(blob[0].total() / 14));
        cv::Mat conf =
            blob[1].reshape(1, static_cast<int>(blob[1].total() / 2));
        cv::Mat iou =
            blob[2].reshape(1, static_cast<int>(blob[2].total() / 1));

        const int num_priors = static_cast<int>(priors_.size());
        if (loc.rows != num_priors || conf.rows != num_priors ||
            iou.rows != num_priors) {
            throw std::runtime_error(cv::format(
                "Unexpected output shape: loc=%d, conf=%d, iou=%d, priors=%d",
                loc.rows, conf.rows, iou.rows, num_priors));
        }

        std::vector<Detection> candidates;
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        candidates.reserve(num_priors);
        boxes.reserve(num_priors);
        scores.reserve(num_priors);

        for (int i = 0; i < num_priors; ++i) {
            const auto& prior = priors_[i];
            const float iou_score =
                std::max(0.0f, std::min(1.0f, iou.at<float>(i, 0)));
            const float cls_score = conf.at<float>(i, 1);
            const float score = std::sqrt(std::max(0.0f, cls_score * iou_score));

            Detection det;
            det.score = score;
            det.corners = {decodePoint(loc, i, 4, prior),
                           decodePoint(loc, i, 6, prior),
                           decodePoint(loc, i, 10, prior),
                           decodePoint(loc, i, 12, prior)};

            std::vector<cv::Point2f> points(det.corners.begin(),
                                            det.corners.end());
            cv::Rect box = cv::boundingRect(points);
            if (box.width <= 0 || box.height <= 0) {
                continue;
            }

            candidates.push_back(det);
            boxes.push_back(box);
            scores.push_back(score);
        }

        std::vector<int> keep_idx;
        cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_,
                          keep_idx, 1.0f, top_k_);

        const int keep_count =
            std::min(static_cast<int>(keep_idx.size()), keep_top_k_);
        std::vector<Detection> results;
        results.reserve(std::max(0, keep_count));
        for (int i = 0; i < keep_count; ++i) {
            results.push_back(candidates[keep_idx[i]]);
        }
        return results;
    }

    cv::Point2f decodePoint(const cv::Mat& loc, int row, int col,
                            const std::array<float, 4>& prior) const {
        const float x =
            (prior[0] + loc.at<float>(row, col) * variance_[0] * prior[2]) *
            static_cast<float>(input_size_.width);
        const float y =
            (prior[1] + loc.at<float>(row, col + 1) * variance_[0] * prior[3]) *
            static_cast<float>(input_size_.height);
        return cv::Point2f(x, y);
    }

    std::string model_path_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    int keep_top_k_;
    int backend_id_;
    int target_id_;
    cv::dnn::Net model_;
    std::vector<std::array<float, 4>> priors_;

    const std::vector<cv::String> output_names_ = {"loc", "conf", "iou"};
    const std::array<std::vector<int>, 4> min_sizes_ = {
        std::vector<int>{10, 16, 24}, std::vector<int>{32, 48},
        std::vector<int>{64, 96}, std::vector<int>{128, 192, 256}};
    const std::array<int, 4> steps_ = {8, 16, 32, 64};
    const std::array<float, 2> variance_ = {0.1f, 0.2f};
};

cv::Mat visualize(const cv::Mat& image, const std::vector<Detection>& results,
                  bool print_results = false, float fps = 0.0f) {
    cv::Mat output = image.clone();
    const cv::Scalar line_color(0, 255, 0);
    const cv::Scalar text_color(0, 0, 255);

    if (fps > 0.0f) {
        cv::putText(output, cv::format("FPS: %.2f", fps), cv::Point(0, 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color);
    }

    for (size_t i = 0; i < results.size(); ++i) {
        const Detection& result = results[i];
        for (size_t j = 0; j < result.corners.size(); ++j) {
            const cv::Point p1(cvRound(result.corners[j].x),
                               cvRound(result.corners[j].y));
            const cv::Point2f& next =
                result.corners[(j + 1) % result.corners.size()];
            const cv::Point p2(cvRound(next.x), cvRound(next.y));
            cv::line(output, p1, p2, line_color, 2);
        }

        const cv::Point text_origin(cvRound(result.corners[0].x),
                                    cvRound(result.corners[0].y) + 12);
        cv::putText(output, cv::format("%.4f", result.score),
                    text_origin, cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);

        if (print_results) {
            std::cout << "-----------license plate " << i + 1
                      << "-----------\n";
            std::cout << "score: " << result.score << "\n";
            std::cout << "corners:";
            for (const auto& point : result.corners) {
                std::cout << " (" << point.x << ", " << point.y << ")";
            }
            std::cout << "\n";
        }
    }

    return output;
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(
        argc, argv,
        "{help h usage ? |      | print this message }"
        "{input i       |      | path to input image. Omit for camera input }"
        "{model m       | license_plate_detection_lpd_yunet_2023mar.onnx | "
        "path to model file }"
        "{backend_target bt | 0 | backend-target pair (0:OpenCV CPU, 1:CUDA, "
        "2:CUDA FP16, 3:TIM-VX NPU, 4:CANN NPU) }"
        "{conf_threshold | 0.9 | minimum confidence threshold }"
        "{nms_threshold  | 0.3 | NMS threshold }"
        "{top_k          | 5000 | keep top_k bounding boxes before NMS }"
        "{keep_top_k     | 750 | keep keep_top_k bounding boxes after NMS }"
        "{save s         |      | save results to result.jpg }"
        "{vis v          |      | visualize results }");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    const int backend_target = parser.get<int>("backend_target");
    if (backend_target < 0 ||
        backend_target >= static_cast<int>(backend_target_pairs.size())) {
        std::cerr << "Error: Invalid backend_target value\n";
        return -1;
    }

    const int backend_id = backend_target_pairs[backend_target].first;
    const int target_id = backend_target_pairs[backend_target].second;

    LPDYuNet model(parser.get<std::string>("model"), cv::Size(320, 240),
                   parser.get<float>("conf_threshold"),
                   parser.get<float>("nms_threshold"),
                   parser.get<int>("top_k"), parser.get<int>("keep_top_k"),
                   backend_id, target_id);

    if (parser.has("input")) {
        const std::string input_path = parser.get<std::string>("input");
        cv::Mat image = cv::imread(input_path);
        if (image.empty()) {
            std::cerr << "Error: Could not read image: " << input_path << "\n";
            return -1;
        }

        model.setInputSize(image.size());
        const std::vector<Detection> results = model.infer(image);
        std::cout << results.size() << " license plates detected.\n";

        if (parser.has("save") || parser.has("vis")) {
            cv::Mat output = visualize(image, results, true);

            if (parser.has("save")) {
                cv::imwrite("result.jpg", output);
                std::cout << "Results saved to result.jpg\n";
            }

            if (parser.has("vis")) {
                cv::namedWindow(input_path, cv::WINDOW_AUTOSIZE);
                cv::imshow(input_path, output);
                cv::waitKey(0);
            }
        }
    } else {
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera\n";
            return -1;
        }

        cv::TickMeter tick_meter;
        cv::Size current_size;
        while (cv::waitKey(1) < 0) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                std::cout << "No frames grabbed!\n";
                break;
            }

            if (frame.size() != current_size) {
                model.setInputSize(frame.size());
                current_size = frame.size();
            }

            tick_meter.start();
            const std::vector<Detection> results = model.infer(frame);
            tick_meter.stop();

            cv::Mat output =
                visualize(frame, results, false,
                          static_cast<float>(tick_meter.getFPS()));
            cv::imshow("LPD-YuNet Demo", output);
            tick_meter.reset();
        }
    }

    return 0;
}
