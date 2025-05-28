#include "opencv2/opencv.hpp"
#include "opencv2/core/types.hpp"

#include <string>
#include <vector>

const std::vector<std::pair<int, int>> backend_target_pairs = {
    {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
    {cv::dnn::DNN_BACKEND_CUDA,   cv::dnn::DNN_TARGET_CUDA},
    {cv::dnn::DNN_BACKEND_CUDA,   cv::dnn::DNN_TARGET_CUDA_FP16},
    {cv::dnn::DNN_BACKEND_TIMVX,  cv::dnn::DNN_TARGET_NPU},
    {cv::dnn::DNN_BACKEND_CANN,   cv::dnn::DNN_TARGET_NPU}
};

class YuNet
{
  public:
    YuNet(const std::string& model_path,
          const cv::Size& input_size,
          const float conf_threshold,
          const float nms_threshold,
          const int top_k,
          const int backend_id,
          const int target_id)
    {
        _detector = cv::FaceDetectorYN::create(
            model_path, "", input_size, conf_threshold, nms_threshold, top_k, backend_id, target_id);
    }

    void setInputSize(const cv::Size& input_size)
    {
        _detector->setInputSize(input_size);
    }

    void setTopK(const int top_k)
    {
        _detector->setTopK(top_k);
    }

    cv::Mat infer(const cv::Mat& image)
    {
        cv::Mat result;
        _detector->detect(image, result);
        return result;
    }

  private:
    cv::Ptr<cv::FaceDetectorYN> _detector;
};

class SFace
{
  public:
    SFace(const std::string& model_path,
          const int backend_id,
          const int target_id,
          const int distance_type)
        : _distance_type(static_cast<cv::FaceRecognizerSF::DisType>(distance_type))
    {
        _recognizer = cv::FaceRecognizerSF::create(model_path, "", backend_id, target_id);
    }

    cv::Mat extractFeatures(const cv::Mat& orig_image, const cv::Mat& face_image)
    {
        // Align and crop detected face from original image
        cv::Mat target_aligned;
        _recognizer->alignCrop(orig_image, face_image, target_aligned);
        // Extract features from cropped detected face
        cv::Mat target_features;
        _recognizer->feature(target_aligned, target_features);
        return target_features.clone();
    }

    std::pair<double, bool> matchFeatures(const cv::Mat& target_features, const cv::Mat& query_features)
    {
        const double score = _recognizer->match(target_features, query_features, _distance_type);
        if (_distance_type == cv::FaceRecognizerSF::DisType::FR_COSINE)
        {
            return {score, score >= _threshold_cosine};
        }
        return {score, score <= _threshold_norml2};
    }

  private:
    cv::Ptr<cv::FaceRecognizerSF> _recognizer;
    cv::FaceRecognizerSF::DisType _distance_type;
    double _threshold_cosine = 0.363;
    double _threshold_norml2 = 1.128;
};

cv::Mat visualize(const cv::Mat& image,
                  const cv::Mat& faces,
                  const std::vector<std::pair<double, bool>>& matches,
                  const float fps = -0.1F,
                  const cv::Size& target_size = cv::Size(512, 512))
{
    static const cv::Scalar matched_box_color{0, 255, 0};
    static const cv::Scalar mismatched_box_color{0, 0, 255};

    if (fps >= 0)
    {
        cv::Mat output_image = image.clone();

        const int x1 = static_cast<int>(faces.at<float>(0, 0));
        const int y1 = static_cast<int>(faces.at<float>(0, 1));
        const int w = static_cast<int>(faces.at<float>(0, 2));
        const int h = static_cast<int>(faces.at<float>(0, 3));
        const auto match = matches.at(0);

        cv::Scalar box_color = match.second ? matched_box_color : mismatched_box_color;
        // Draw bounding box
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);
        // Draw match score
        cv::putText(output_image, cv::format("%.4f", match.first), cv::Point(x1, y1+12), cv::FONT_HERSHEY_DUPLEX, 0.30, box_color);
        // Draw FPS
        cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2);

        return output_image;
    }

    cv::Mat output_image = cv::Mat::zeros(target_size, CV_8UC3);

    // Determine new height and width of image with aspect ratio of original image
    const double ratio = std::min(static_cast<double>(target_size.height) / image.rows,
                                  static_cast<double>(target_size.width) / image.cols);
    const int new_height = static_cast<int>(image.rows * ratio);
    const int new_width = static_cast<int>(image.cols * ratio);

    // Resize the original image, maintaining aspect ratio
    cv::Mat resize_out;
    cv::resize(image, resize_out, cv::Size(new_width, new_height), cv::INTER_LINEAR);

    // Determine top left corner in resized dimensions
    const int top = std::max(0, target_size.height - new_height) / 2;
    const int left = std::max(0, target_size.width - new_width) / 2;

    // Copy resized image into target output image
    const cv::Rect roi = cv::Rect(cv::Point(left, top), cv::Size(new_width, new_height));
    cv::Mat out_sub_image = output_image(roi);
    resize_out.copyTo(out_sub_image);

    for (int i = 0; i < faces.rows; ++i)
    {
        const int x1 = static_cast<int>(faces.at<float>(i, 0) * ratio) + left;
        const int y1 = static_cast<int>(faces.at<float>(i, 1) * ratio) + top;
        const int w = static_cast<int>(faces.at<float>(i, 2) * ratio);
        const int h = static_cast<int>(faces.at<float>(i, 3) * ratio);
        const auto match = matches.at(i);

        cv::Scalar box_color = match.second ? matched_box_color : mismatched_box_color;
        // Draw bounding box
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);
        // Draw match score
        cv::putText(output_image, cv::format("%.4f", match.first), cv::Point(x1, y1+12), cv::FONT_HERSHEY_DUPLEX, 0.30, box_color);
    }
    return output_image;
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        // General options
        "{help  h           |                                     | Print this message}"
        "{backend_target b  | 0                                   | Set DNN backend target pair:\n"
                                                                   "0: (default) OpenCV implementation + CPU,\n"
                                                                   "1: CUDA + GPU (CUDA),\n"
                                                                   "2: CUDA + GPU (CUDA FP16),\n"
                                                                   "3: TIM-VX + NPU,\n"
                                                                   "4: CANN + NPU}"
        "{save s            | false                               | Whether to save result image or not}"
        "{vis v             | false                               | Whether to visualize result image or not}"
        // SFace options
        "{target_face t     |                                     | Set path to input image 1 (target face)}"
        "{query_face q      |                                     | Set path to input image 2 (query face), omit if using camera}"
        "{model m           | face_recognition_sface_2021dec.onnx | Set path to the model}"
        "{distance_type d   | 0                                   | 0 = cosine, 1 = norm_l1}"
        // YuNet options
        "{yunet_model       | ../face_detection_yunet/face_detection_yunet_2023mar.onnx | Set path to the YuNet model}"
        "{detect_threshold  | 0.9                                                       | Set the minimum confidence for the model\n"
                                                                                         "to identify a face. Filter out faces of\n"
                                                                                         "conf < conf_threshold}"
        "{nms_threshold     | 0.3                                                       | Set the threshold to suppress overlapped boxes.\n"
                                                                                         "Suppress boxes if IoU(box1, box2) >= nms_threshold\n"
                                                                                         ", the one of higher score is kept.}"
        "{top_k             | 5000                                                      | Keep top_k bounding boxes before NMS}"
    );

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // General CLI options
    const int backend = parser.get<int>("backend_target");
    const bool save_flag = parser.get<bool>("save");
    const bool vis_flag = parser.get<bool>("vis");
    const int backend_id = backend_target_pairs.at(backend).first;
    const int target_id = backend_target_pairs.at(backend).second;

    // YuNet CLI options
    const std::string detector_model_path = parser.get<std::string>("yunet_model");
    const float detect_threshold = parser.get<float>("detect_threshold");
    const float nms_threshold = parser.get<float>("nms_threshold");
    const int top_k = parser.get<int>("top_k");

    // Use YuNet as the detector backend
    auto face_detector = YuNet(
        detector_model_path, cv::Size(320, 320), detect_threshold, nms_threshold, top_k, backend_id, target_id);

    // SFace CLI options
    const std::string target_path = parser.get<std::string>("target_face");
    const std::string query_path = parser.get<std::string>("query_face");
    const std::string model_path = parser.get<std::string>("model");
    const int distance_type = parser.get<int>("distance_type");

    auto face_recognizer = SFace(model_path, backend_id, target_id, distance_type);

    if (target_path.empty())
    {
        CV_Error(cv::Error::StsError, "Path to target image " + target_path + " not found");
    }

    cv::Mat target_image = cv::imread(target_path);
    // Detect single face in target image
    face_detector.setInputSize(target_image.size());
    face_detector.setTopK(1);
    cv::Mat target_face = face_detector.infer(target_image);
    // Extract features from target face
    cv::Mat target_features = face_recognizer.extractFeatures(target_image, target_face.row(0));

    if (!query_path.empty()) // use image
    {
        // Detect any faces in query image
        cv::Mat query_image = cv::imread(query_path);
        face_detector.setInputSize(query_image.size());
        face_detector.setTopK(5000);
        cv::Mat query_faces = face_detector.infer(query_image);

        // Store match scores for visualization
        std::vector<std::pair<double, bool>> matches;

        for (int i = 0; i < query_faces.rows; ++i)
        {
            // Extract features from query face
            cv::Mat query_features = face_recognizer.extractFeatures(query_image, query_faces.row(i));
            // Measure similarity of target face to query face
            const auto match = face_recognizer.matchFeatures(target_features, query_features);
            matches.push_back(match);

            const int x1 = static_cast<int>(query_faces.at<float>(i, 0));
            const int y1 = static_cast<int>(query_faces.at<float>(i, 1));
            const int w = static_cast<int>(query_faces.at<float>(i, 2));
            const int h = static_cast<int>(query_faces.at<float>(i, 3));
            const float conf = query_faces.at<float>(i, 14);

            std::cout << cv::format("%d: x1=%d, y1=%d, w=%d, h=%d, conf=%.4f, match=%.4f\n", i, x1, y1, w, h, conf, match.first);
        }

        if (save_flag || vis_flag)
        {
            auto vis_target = visualize(target_image, target_face, {{1.0, true}});
            auto vis_query = visualize(query_image, query_faces, matches);
            cv::Mat output_image;
            cv::hconcat(vis_target, vis_query, output_image);

            if (save_flag)
            {
                std::cout << "Results are saved to result.jpg\n";
                cv::imwrite("result.jpg", output_image);
            }
            if (vis_flag)
            {
                cv::namedWindow(query_path, cv::WINDOW_AUTOSIZE);
                cv::imshow(query_path, output_image);
                cv::waitKey(0);
            }
        }
    }
    else // use video capture
    {
        const int device_id = 0;
        auto cap = cv::VideoCapture(device_id);
        const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        face_detector.setInputSize(cv::Size(w, h));

        auto tick_meter = cv::TickMeter();
        cv::Mat query_frame;

        while (cv::waitKey(1) < 0)
        {
            bool has_frame = cap.read(query_frame);
            if (!has_frame)
            {
                std::cout << "No frames grabbed! Exiting ...\n";
                break;
            }
            tick_meter.start();
            // Detect faces from webcam image
            cv::Mat query_faces = face_detector.infer(query_frame);
            tick_meter.stop();

            // Extract features from query face
            cv::Mat query_features = face_recognizer.extractFeatures(query_frame, query_faces.row(0));
            // Measure similarity of target face to query face
            const auto match = face_recognizer.matchFeatures(target_features, query_features);

            const auto fps = static_cast<float>(tick_meter.getFPS());

            auto vis_target = visualize(target_image, target_face, {{1.0, true}}, -0.1F, cv::Size(w, h));
            auto vis_query = visualize(query_frame, query_faces, {match}, fps);
            cv::Mat output_image;
            cv::hconcat(vis_target, vis_query, output_image);

            // Visualize in a new window
            cv::imshow("SFace Demo", output_image);

            tick_meter.reset();
        }
    }
    return 0;
}
