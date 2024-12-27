#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

const std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>>
    backend_target_pairs = {
        {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16},
        {cv::dnn::DNN_BACKEND_TIMVX, cv::dnn::DNN_TARGET_NPU},
        {cv::dnn::DNN_BACKEND_CANN, cv::dnn::DNN_TARGET_NPU}};

class MPPalmDet {
   private:
    std::string model_path;
    float nms_threshold;
    float score_threshold;
    int topK;
    int backend_id;
    int target_id;
    cv::Size input_size;
    cv::dnn::Net model;
    std::vector<cv::Point2f> anchors;

    std::vector<cv::Point2f> loadAnchors() {
        std::vector<cv::Point2f> anchors = {
            {0.02083333f, 0.02083333f}, {0.02083333f, 0.02083333f},
            {0.0625f, 0.02083333f},     {0.0625f, 0.02083333f},
            {0.10416666f, 0.02083333f}, {0.10416666f, 0.02083333f},
            {0.14583333f, 0.02083333f}, {0.14583333f, 0.02083333f},
        }; // Change this to all anchors
        return anchors;
    }

    std::pair<cv::Mat, cv::Point2i> preprocess(const cv::Mat& image) {
        cv::Point2i pad_bias(0, 0);
        float ratio =
            std::min(static_cast<float>(input_size.width) / image.cols,
                     static_cast<float>(input_size.height) / image.rows);

        cv::Mat processed_image;
        if (image.rows != input_size.height || image.cols != input_size.width) {
            cv::Size ratio_size(static_cast<int>(image.cols * ratio),
                                static_cast<int>(image.rows * ratio));
            cv::resize(image, processed_image, ratio_size);

            int pad_h = input_size.height - ratio_size.height;
            int pad_w = input_size.width - ratio_size.width;
            pad_bias.x = pad_w / 2;
            pad_bias.y = pad_h / 2;

            cv::copyMakeBorder(processed_image, processed_image, pad_bias.y,
                               pad_h - pad_bias.y, pad_bias.x,
                               pad_w - pad_bias.x, cv::BORDER_CONSTANT,
                               cv::Scalar(0, 0, 0));
        } else {
            processed_image = image.clone();
        }

        // Create blob with correct parameters
        cv::Mat blob;
        cv::dnn::Image2BlobParams params;
        params.datalayout = cv::dnn::DNN_LAYOUT_NHWC;
        params.ddepth = CV_32F;
        params.mean = cv::Scalar::all(0);
        params.scalefactor = cv::Scalar::all(1.0 / 255.0);
        params.size = input_size;
        params.swapRB = true;
        params.paddingmode = cv::dnn::DNN_PMODE_NULL;

        blob = cv::dnn::blobFromImageWithParams(processed_image, params);

        pad_bias.x = static_cast<int>(pad_bias.x / ratio);
        pad_bias.y = static_cast<int>(pad_bias.y / ratio);

        return {blob, pad_bias};
    }

    std::vector<std::vector<float>> postprocess(
        const std::vector<cv::Mat>& output_blobs, const cv::Size& original_size,
        const cv::Point2i& pad_bias) {
        
        cv::Mat scores = output_blobs[1].reshape(1, output_blobs[1].total() / 1);
        cv::Mat boxes = output_blobs[0].reshape(1, output_blobs[0].total() / 18);

        std::vector<float> score_vec;
        std::vector<cv::Rect2f> boxes_vec;
        std::vector<std::vector<cv::Point2f>> landmarks_vec;

        // Match Python's scale calculation exactly
        float scale = std::max(original_size.height, original_size.width);

        // Process all detections first, like Python
        for (int i = 0; i < scores.rows; i++) {
            float score = 1.0f / (1.0f + std::exp(-scores.at<float>(i, 0)));
            
            // Extract box and landmark deltas
            cv::Mat box_delta = boxes.row(i).colRange(0, 4);
            cv::Mat landmark_delta = boxes.row(i).colRange(4, 18);
            cv::Point2f anchor = anchors[i];

            // Normalize box deltas by input size
            cv::Point2f cxy_delta(box_delta.at<float>(0) / input_size.width,
                                 box_delta.at<float>(1) / input_size.height);
            cv::Point2f wh_delta(box_delta.at<float>(2) / input_size.width,
                                box_delta.at<float>(3) / input_size.height);

            // Calculate box coordinates (scale first, then subtract pad_bias)
            cv::Point2f xy1((cxy_delta.x - wh_delta.x / 2 + anchor.x) * scale - pad_bias.x,
                           (cxy_delta.y - wh_delta.y / 2 + anchor.y) * scale - pad_bias.y);
            cv::Point2f xy2((cxy_delta.x + wh_delta.x / 2 + anchor.x) * scale - pad_bias.x,
                           (cxy_delta.y + wh_delta.y / 2 + anchor.y) * scale - pad_bias.y);

            if (score > score_threshold) {
                score_vec.push_back(score);
                boxes_vec.push_back(cv::Rect2f(xy1.x, xy1.y, xy2.x - xy1.x, xy2.y - xy1.y));

                // Process landmarks
                std::vector<cv::Point2f> landmarks;
                for (int j = 0; j < 7; j++) {
                    // Normalize by input size
                    float dx = landmark_delta.at<float>(j * 2) / input_size.width;
                    float dy = landmark_delta.at<float>(j * 2 + 1) / input_size.height;
                    
                    // Add anchor
                    dx += anchor.x;
                    dy += anchor.y;
                    
                    // Scale and subtract pad_bias in one step
                    dx = dx * scale - pad_bias.x;
                    dy = dy * scale - pad_bias.y;
                    
                    landmarks.push_back(cv::Point2f(dx, dy));
                }
                landmarks_vec.push_back(landmarks);
            }
        }

        // Perform NMS
        std::vector<int> indices;
        std::vector<cv::Rect> boxes_int;
        for (const auto& box : boxes_vec) {
            boxes_int.push_back(cv::Rect(
                static_cast<int>(box.x), static_cast<int>(box.y),
                static_cast<int>(box.width), static_cast<int>(box.height)));
        }
        cv::dnn::NMSBoxes(boxes_int, score_vec, score_threshold, nms_threshold, indices);

        // Prepare results
        std::vector<std::vector<float>> results;
        for (int idx : indices) {
            std::vector<float> result;
            result.push_back(boxes_vec[idx].x);
            result.push_back(boxes_vec[idx].y);
            result.push_back(boxes_vec[idx].x + boxes_vec[idx].width);
            result.push_back(boxes_vec[idx].y + boxes_vec[idx].height);

            for (const auto& point : landmarks_vec[idx]) {
                result.push_back(point.x);
                result.push_back(point.y);
            }
            result.push_back(score_vec[idx]);
            results.push_back(result);
        }

        return results;
    }

   public:
    MPPalmDet(const std::string& modelPath, float nmsThreshold = 0.3f,
              float scoreThreshold = 0.5f, int topK = 5000,
              int backendId = cv::dnn::DNN_BACKEND_DEFAULT,
              int targetId = cv::dnn::DNN_TARGET_CPU)
        : model_path(modelPath),
          nms_threshold(nmsThreshold),
          score_threshold(scoreThreshold),
          topK(topK),
          backend_id(backendId),
          target_id(targetId),
          input_size(192, 192) {
        model = cv::dnn::readNet(model_path);
        model.setPreferableBackend(backend_id);
        model.setPreferableTarget(target_id);
        anchors = loadAnchors();
    }

    void setBackendAndTarget(int backendId, int targetId) {
        backend_id = backendId;
        target_id = targetId;
        model.setPreferableBackend(backend_id);
        model.setPreferableTarget(target_id);
    }

    std::vector<std::vector<float>> infer(const cv::Mat& image) {
        std::pair<cv::Mat, cv::Point2i> preprocess_result = preprocess(image);
        cv::Mat preprocessed_image = preprocess_result.first;
        cv::Point2i pad_bias = preprocess_result.second;
        model.setInput(preprocessed_image);
        std::vector<cv::Mat> outputs;
        model.forward(outputs, model.getUnconnectedOutLayersNames());
        return postprocess(outputs, image.size(), pad_bias);
    }
};

class HandDetectorDemo {
   private:
    MPPalmDet detector;

    cv::Mat visualize(const cv::Mat& image,
                      const std::vector<std::vector<float>>& results,
                      bool print_results = false, float fps = 0.0f) {
        cv::Mat output = image.clone();

        if (fps > 0) {
            cv::putText(output, cv::format("FPS: %.2f", fps), cv::Point(0, 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        }

        for (size_t i = 0; i < results.size(); i++) {
            const std::vector<float>& result = results[i];
            float score = result.back();

            // Draw box - using direct coordinates like Python version
            cv::rectangle(output, 
                cv::Point(static_cast<int>(result[0]), static_cast<int>(result[1])),
                cv::Point(static_cast<int>(result[2]), static_cast<int>(result[3])),
                cv::Scalar(0, 255, 0), 2);

            // Put score - using first coordinate of box
            cv::putText(output, cv::format("%.4f", score),
                        cv::Point(static_cast<int>(result[0]), 
                                 static_cast<int>(result[1]) + 12),
                        cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0));

            // Draw landmarks
            for (size_t j = 0; j < 7; j++) {
                cv::Point point(static_cast<int>(result[4 + j * 2]),
                              static_cast<int>(result[4 + j * 2 + 1]));
                cv::circle(output, point, 2, cv::Scalar(0, 0, 255), 2);
            }

            if (print_results) {
                std::cout << "-----------palm " << i + 1 << "-----------\n";
                std::cout << "score: " << score << "\n";
                std::cout << "palm box: [" 
                          << result[0] << ", " 
                          << result[1] << ", "
                          << result[2] << ", "
                          << result[3] << "]\n";
                std::cout << "palm landmarks:\n";
                for (size_t j = 0; j < 7; j++) {
                    std::cout << "\t(" << result[4 + j * 2] << ", "
                              << result[4 + j * 2 + 1] << ")\n";
                }
            }
        }

        return output;
    }

   public:
    HandDetectorDemo(const std::string& model_path, float nms_threshold = 0.3f,
                     float score_threshold = 0.8f,
                     int backend_id = cv::dnn::DNN_BACKEND_DEFAULT,
                     int target_id = cv::dnn::DNN_TARGET_CPU)
        : detector(model_path, nms_threshold, score_threshold, 5000, backend_id,
                   target_id) {}

    void processImage(const std::string& input_path, bool save = false,
                      bool vis = false) {
        cv::Mat image = cv::imread(input_path);
        if (image.empty()) {
            std::cerr << "Error: Could not read image: " << input_path
                      << std::endl;
            return;
        }

        std::vector<std::vector<float>> results = detector.infer(image);
        if (results.empty()) {
            std::cout << "Hand not detected" << std::endl;
        }

        cv::Mat output = visualize(image, results, true);

        if (save) {
            cv::imwrite("result.jpg", output);
            std::cout << "Results saved to result.jpg\n" << std::endl;
        }

        if (vis) {
            cv::namedWindow(input_path, cv::WINDOW_AUTOSIZE);
            cv::imshow(input_path, output);
            cv::waitKey(0);
        }
    }

    void processCamera(int device_id = 0) {
        cv::VideoCapture cap(device_id);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera" << std::endl;
            return;
        }

        std::chrono::steady_clock::time_point start_time =
            std::chrono::steady_clock::now();
        int frame_count = 0;

        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty()) {
                std::cout << "No frames grabbed!" << std::endl;
                break;
            }

            std::vector<std::vector<float>> results = detector.infer(frame);
            frame_count++;

            std::chrono::steady_clock::time_point current_time =
                std::chrono::steady_clock::now();
            float fps =
                frame_count / (std::chrono::duration_cast<std::chrono::seconds>(
                                   current_time - start_time)
                                   .count() +
                               1);

            cv::Mat output = visualize(frame, results, false, fps);
            cv::imshow("MPPalmDet Demo", output);

            if (cv::waitKey(1) >= 0) break;
        }
    }
};

int main(int argc, char** argv) {
    cv::CommandLineParser parser(
        argc, argv,
        "{help h usage ? |      | print this message }"
        "{input i       |      | path to input image }"
        "{model m       | palm_detection_mediapipe_2023feb.onnx | path to "
        "model file }"
        "{backend_target bt | 0 | backend-target pair (0:OpenCV CPU, 1:CUDA, "
        "2:CUDA FP16, 3:TIM-VX NPU, 4:CANN NPU) }"
        "{score_threshold | 0.8 | minimum confidence threshold }"
        "{nms_threshold   | 0.3 | NMS threshold }"
        "{save s         |     | save results to file }"
        "{vis v         |     | visualize results }");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    int backend_target = parser.get<int>("backend_target");
    if (backend_target < 0 || backend_target >= backend_target_pairs.size()) {
        std::cerr << "Error: Invalid backend_target value" << std::endl;
        return -1;
    }

    int backend_id = backend_target_pairs[backend_target].first;
    int target_id = backend_target_pairs[backend_target].second;

    HandDetectorDemo demo(
        parser.get<std::string>("model"), parser.get<float>("nms_threshold"),
        parser.get<float>("score_threshold"), backend_id, target_id);

    if (parser.has("input")) {
        demo.processImage(parser.get<std::string>("input"), parser.has("save"),
                          parser.has("vis"));
    } else {
        demo.processCamera();
    }

    return 0;
}