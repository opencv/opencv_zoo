// This file is part of OpenCV Zoo project.
// It is subject to the license terms in the LICENSE file found in the same directory.
//
// Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>
#include "mp_palmdet.hpp"

struct HandPose {
    cv::Rect2f bbox;
    std::vector<cv::Point3f> landmarks;
    std::vector<cv::Point3f> world_landmarks;
    float handedness; // < 0.5 Left, >= 0.5 Right
    float conf;
    std::string gesture;
};

class MPHandPose {
private:
    std::string model_path;
    float conf_threshold;
    int backend_id;
    int target_id;
    cv::Size input_size;
    cv::dnn::Net model;

    const int PALM_BASE = 0;
    const int MIDDLE_BASE = 2;
    const cv::Point2f PALM_BOX_PRE_SHIFT = cv::Point2f(0.0f, 0.0f);
    const float PALM_BOX_PRE_ENLARGE = 4.0f;
    const cv::Point2f PALM_BOX_SHIFT = cv::Point2f(0.0f, -0.4f);
    const float PALM_BOX_ENLARGE = 3.0f;
    const cv::Point2f HAND_BOX_SHIFT = cv::Point2f(0.0f, -0.1f);
    const float HAND_BOX_ENLARGE = 1.65f;

    std::tuple<cv::Mat, cv::Rect2f, cv::Point2i> cropAndPadFromPalm(
        const cv::Mat& image, cv::Rect2f palm_bbox, bool for_rotation = false) {
        cv::Point2f shift_vector = for_rotation ? PALM_BOX_PRE_SHIFT : PALM_BOX_SHIFT;
        cv::Point2f wh(palm_bbox.width, palm_bbox.height);
        palm_bbox.x += shift_vector.x * wh.x;
        palm_bbox.y += shift_vector.y * wh.y;

        float enlarge_scale = for_rotation ? PALM_BOX_PRE_ENLARGE : PALM_BOX_ENLARGE;
        cv::Point2f center(palm_bbox.x + palm_bbox.width / 2.0f, palm_bbox.y + palm_bbox.height / 2.0f);
        cv::Point2f half_size(wh.x * enlarge_scale / 2.0f, wh.y * enlarge_scale / 2.0f);

        float x1 = std::max(0.0f, std::min(static_cast<float>(image.cols), center.x - half_size.x));
        float y1 = std::max(0.0f, std::min(static_cast<float>(image.rows), center.y - half_size.y));
        float x2 = std::max(0.0f, std::min(static_cast<float>(image.cols), center.x + half_size.x));
        float y2 = std::max(0.0f, std::min(static_cast<float>(image.rows), center.y + half_size.y));

        cv::Rect roi(cvRound(x1), cvRound(y1), cvRound(x2 - x1), cvRound(y2 - y1));
        cv::Mat cropped = image(roi).clone();

        int side_len = for_rotation
            ? static_cast<int>(std::sqrt(image.rows * image.rows + image.cols * image.cols))
            : std::max(cropped.rows, cropped.cols);

        int pad_h = side_len - cropped.rows;
        int pad_w = side_len - cropped.cols;
        int left = pad_w / 2;
        int top = pad_h / 2;
        int right = pad_w - left;
        int bottom = pad_h - top;

        cv::Mat padded;
        cv::copyMakeBorder(cropped, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Point2i pad_bias(static_cast<int>(x1) - left, static_cast<int>(y1) - top);
        return {padded, cv::Rect2f(x1, y1, x2 - x1, y2 - y1), pad_bias};
    }

    std::tuple<cv::Mat, cv::Rect2f, float, cv::Mat, cv::Point2i> preprocess(
        const cv::Mat& image, const std::vector<float>& palm) {
        cv::Rect2f palm_bbox(palm[0], palm[1], palm[2] - palm[0], palm[3] - palm[1]);
        auto [padded_img, cropped_bbox, bias] = cropAndPadFromPalm(image, palm_bbox, true);

        cv::Mat rgb_img;
        cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);

        cv::Point2f p1(palm[4 + PALM_BASE * 2] - bias.x, palm[4 + PALM_BASE * 2 + 1] - bias.y);
        cv::Point2f p2(palm[4 + MIDDLE_BASE * 2] - bias.x, palm[4 + MIDDLE_BASE * 2 + 1] - bias.y);

        float radians = CV_PI / 2.0f - std::atan2(-(p2.y - p1.y), p2.x - p1.x);
        radians = radians - 2.0f * CV_PI * std::floor((radians + CV_PI) / (2.0f * CV_PI));
        float angle = radians * 180.0f / CV_PI;

        cv::Point2f center(cropped_bbox.x - bias.x + cropped_bbox.width / 2.0f,
                           cropped_bbox.y - bias.y + cropped_bbox.height / 2.0f);

        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::Mat rotated_img;
        cv::warpAffine(rgb_img, rotated_img, rot_mat, rgb_img.size());

        std::vector<cv::Point2f> palm_landmarks(7);
        for (int i = 0; i < 7; ++i) {
            palm_landmarks[i] = cv::Point2f(palm[4 + i * 2] - bias.x, palm[4 + i * 2 + 1] - bias.y);
        }
        std::vector<cv::Point2f> rotated_landmarks;
        cv::transform(palm_landmarks, rotated_landmarks, rot_mat);

        float min_x = rotated_landmarks[0].x, max_x = rotated_landmarks[0].x;
        float min_y = rotated_landmarks[0].y, max_y = rotated_landmarks[0].y;
        for (const auto& pt : rotated_landmarks) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
        }
        cv::Rect2f rotated_palm_bbox(min_x, min_y, max_x - min_x, max_y - min_y);

        auto [hand_crop, final_bbox, dummy_bias] = cropAndPadFromPalm(rotated_img, rotated_palm_bbox, false);

        cv::Mat blob_f;
        cv::resize(hand_crop, blob_f, input_size, 0, 0, cv::INTER_AREA);
        blob_f.convertTo(blob_f, CV_32FC3, 1.0 / 255.0);

        int sz[] = {1, input_size.height, input_size.width, 3};
        cv::Mat input_blob(4, sz, CV_32F);
        std::memcpy(input_blob.data, blob_f.data, input_size.height * input_size.width * 3 * sizeof(float));

        return {input_blob, rotated_palm_bbox, angle, rot_mat, bias};
    }

    bool postprocess(const std::vector<cv::Mat>& output_blobs,
                     const cv::Rect2f& rotated_palm_bbox, float angle,
                     const cv::Mat& rot_mat, const cv::Point2i& pad_bias,
                     HandPose& out_hand) {
        cv::Mat landmarks_mat = output_blobs[0];
        cv::Mat conf_mat = output_blobs[1];
        cv::Mat handedness_mat = output_blobs[2];
        cv::Mat world_mat = output_blobs[3];

        float conf = conf_mat.at<float>(0, 0);
        if (conf < conf_threshold) {
            return false;
        }

        out_hand.conf = conf;
        out_hand.handedness = handedness_mat.at<float>(0, 0);

        float wh_scale = std::max(rotated_palm_bbox.width / input_size.width,
                                  rotated_palm_bbox.height / input_size.height);

        std::vector<cv::Point3f> raw_lm(21);
        std::vector<cv::Point3f> raw_world(21);
        const float* lm_ptr = landmarks_mat.ptr<float>(0);
        const float* world_ptr = world_mat.ptr<float>(0);

        for (int i = 0; i < 21; ++i) {
            raw_lm[i] = cv::Point3f((lm_ptr[i * 3] - input_size.width / 2.0f) * wh_scale,
                                   (lm_ptr[i * 3 + 1] - input_size.height / 2.0f) * wh_scale,
                                   lm_ptr[i * 3 + 2] * wh_scale);
            raw_world[i] = cv::Point3f(world_ptr[i * 3], world_ptr[i * 3 + 1], world_ptr[i * 3 + 2]);
        }

        cv::Mat coords_rot = cv::getRotationMatrix2D(cv::Point2f(0, 0), angle, 1.0);
        cv::Mat inv_rot_mat;
        cv::invertAffineTransform(rot_mat, inv_rot_mat);

        cv::Point2f center_rot(rotated_palm_bbox.x + rotated_palm_bbox.width / 2.0f,
                               rotated_palm_bbox.y + rotated_palm_bbox.height / 2.0f);
        cv::Point2f orig_center(
            inv_rot_mat.at<double>(0, 0) * center_rot.x + inv_rot_mat.at<double>(0, 1) * center_rot.y + inv_rot_mat.at<double>(0, 2),
            inv_rot_mat.at<double>(1, 0) * center_rot.x + inv_rot_mat.at<double>(1, 1) * center_rot.y + inv_rot_mat.at<double>(1, 2)
        );

        out_hand.landmarks.resize(21);
        out_hand.world_landmarks.resize(21);

        float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;

        for (int i = 0; i < 21; ++i) {
            float rx = coords_rot.at<double>(0, 0) * raw_lm[i].x + coords_rot.at<double>(0, 1) * raw_lm[i].y;
            float ry = coords_rot.at<double>(1, 0) * raw_lm[i].x + coords_rot.at<double>(1, 1) * raw_lm[i].y;

            float fx = rx + orig_center.x + pad_bias.x;
            float fy = ry + orig_center.y + pad_bias.y;
            float fz = raw_lm[i].z;

            out_hand.landmarks[i] = cv::Point3f(fx, fy, fz);

            min_x = std::min(min_x, fx);
            max_x = std::max(max_x, fx);
            min_y = std::min(min_y, fy);
            max_y = std::max(max_y, fy);

            float wx = coords_rot.at<double>(0, 0) * raw_world[i].x + coords_rot.at<double>(0, 1) * raw_world[i].y;
            float wy = coords_rot.at<double>(1, 0) * raw_world[i].x + coords_rot.at<double>(1, 1) * raw_world[i].y;
            out_hand.world_landmarks[i] = cv::Point3f(wx, wy, raw_world[i].z);
        }

        float bw = max_x - min_x;
        float bh = max_y - min_y;
        float cx = min_x + bw / 2.0f + HAND_BOX_SHIFT.x * bw;
        float cy = min_y + bh / 2.0f + HAND_BOX_SHIFT.y * bh;
        float half_w = bw * HAND_BOX_ENLARGE / 2.0f;
        float half_h = bh * HAND_BOX_ENLARGE / 2.0f;

        out_hand.bbox = cv::Rect2f(cx - half_w, cy - half_h, half_w * 2.0f, half_h * 2.0f);
        out_hand.gesture = classify(out_hand.landmarks);

        return true;
    }

    float computeAngle(const cv::Point3f& p1, const cv::Point3f& p2, const cv::Point3f& p3) {
        cv::Point2f v1(p1.x - p2.x, p1.y - p2.y);
        cv::Point2f v2(p3.x - p2.x, p3.y - p2.y);
        float dot = v1.x * v2.x + v1.y * v2.y;
        float len1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
        float len2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);
        if (len1 * len2 == 0.0f) return 0.0f;
        float cos_angle = std::max(-1.0f, std::min(1.0f, dot / (len1 * len2)));
        return std::acos(cos_angle) * 180.0f / CV_PI;
    }

public:
    MPHandPose(const std::string& modelPath, float confThreshold = 0.8f,
               int backendId = cv::dnn::DNN_BACKEND_DEFAULT,
               int targetId = cv::dnn::DNN_TARGET_CPU)
        : model_path(modelPath),
          conf_threshold(confThreshold),
          backend_id(backendId),
          target_id(targetId),
          input_size(224, 224) {
        model = cv::dnn::readNet(model_path);
        model.setPreferableBackend(backend_id);
        model.setPreferableTarget(target_id);
    }

    bool infer(const cv::Mat& image, const std::vector<float>& palm, HandPose& out_hand) {
        auto [input_blob, rotated_palm_bbox, angle, rot_mat, pad_bias] = preprocess(image, palm);
        model.setInput(input_blob);
        std::vector<cv::Mat> output_blobs;
        model.forward(output_blobs, model.getUnconnectedOutLayersNames());
        return postprocess(output_blobs, rotated_palm_bbox, angle, rot_mat, pad_bias, out_hand);
    }

    std::string classify(const std::vector<cv::Point3f>& lm) {
        float angle_thumb = computeAngle(lm[0], lm[2], lm[4]);
        float angle_index = computeAngle(lm[0], lm[6], lm[8]);
        float angle_middle = computeAngle(lm[0], lm[10], lm[12]);
        float angle_ring = computeAngle(lm[0], lm[14], lm[16]);
        float angle_pinky = computeAngle(lm[0], lm[18], lm[20]);

        float thr = 65.0f;
        float thr_s = 49.0f;

        bool thumb = angle_thumb < thr_s;
        bool index = angle_index < thr;
        bool middle = angle_middle < thr;
        bool ring = angle_ring < thr;
        bool pinky = angle_pinky < thr_s;

        if (!thumb && !index && !middle && !ring && !pinky) return "Fist";
        if (!thumb && index && !middle && !ring && !pinky) return "One";
        if (!thumb && index && middle && !ring && !pinky) return "Two";
        if (!thumb && index && middle && ring && !pinky) return "Three";
        if (!thumb && index && middle && ring && pinky) return "Four";
        if (thumb && index && middle && ring && pinky) return "Five";
        if (thumb && !index && !middle && !ring && pinky) return "Six";
        if (thumb && index && !middle && !ring && !pinky) return "Seven";
        if (thumb && index && middle && !ring && !pinky) return "Eight";
        if (thumb && index && middle && ring && !pinky) return "Nine";
        return "Unknown";
    }
};

cv::Mat visualize(const cv::Mat& image, const std::vector<HandPose>& hands, bool print_results = false) {
    cv::Mat output = image.clone();

    const std::vector<std::pair<int, int>> connections = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4},       // Thumb
        {0, 5}, {5, 6}, {6, 7}, {7, 8},       // Index
        {5, 9}, {9, 10}, {10, 11}, {11, 12},  // Middle
        {9, 13}, {13, 14}, {14, 15}, {15, 16},// Ring
        {13, 17}, {17, 18}, {18, 19}, {19, 20}, // Pinky
        {0, 17}                                // Palm base
    };

    for (size_t h = 0; h < hands.size(); ++h) {
        const auto& hand = hands[h];

        for (const auto& [i, j] : connections) {
            cv::Point p1(cvRound(hand.landmarks[i].x), cvRound(hand.landmarks[i].y));
            cv::Point p2(cvRound(hand.landmarks[j].x), cvRound(hand.landmarks[j].y));
            cv::line(output, p1, p2, cv::Scalar(255, 255, 255), 2);
        }

        for (const auto& pt : hand.landmarks) {
            cv::circle(output, cv::Point(cvRound(pt.x), cvRound(pt.y)), 3, cv::Scalar(0, 0, 255), -1);
        }

        cv::rectangle(output, hand.bbox, cv::Scalar(0, 255, 0), 2);

        std::string label = cv::format("%s (%s, %.2f)",
            hand.gesture.c_str(),
            (hand.handedness >= 0.5f ? "Right" : "Left"),
            hand.conf);
        cv::putText(output, label, cv::Point(cvRound(hand.bbox.x), cvRound(hand.bbox.y) - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        if (print_results) {
            std::cout << "-----------hand " << h + 1 << "-----------\n";
            std::cout << "conf: " << cv::format("%.2f", hand.conf) << "\n";
            std::cout << "handedness: " << (hand.handedness >= 0.5f ? "Right" : "Left") << "\n";
            std::cout << "gesture: " << hand.gesture << "\n";
            std::cout << "hand box: [" << cvRound(hand.bbox.x) << " " << cvRound(hand.bbox.y)
                      << " " << cvRound(hand.bbox.width) << " " << cvRound(hand.bbox.height) << "]\n";
            std::cout << "hand landmarks:\n";
            for (const auto& pt : hand.landmarks) {
                std::cout << "\t[" << cvRound(pt.x) << " " << cvRound(pt.y) << " " << cvRound(pt.z) << "]\n";
            }
        }
    }

    return output;
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(
        argc, argv,
        "{help h usage ? |      | print this message }"
        "{input i       |      | path to input image. Omit for camera. }"
        "{model m       | handpose_estimation_mediapipe_2023feb.onnx | path to handpose model }"
        "{palm_model pm | ../palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx | path to palm detection model }"
        "{backend_target bt | 0 | 0: OpenCV+CPU, 1: CUDA+GPU, 2: CUDA+GPU FP16, 3: TIM-VX, 4: CANN }"
        "{conf_threshold ct | 0.9 | filter out hands with conf < conf_threshold }"
        "{save s        | false | specify to save results to result.jpg }"
        "{vis v         | false | specify to open a window for visualization }");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string input_path = parser.get<std::string>("input");
    std::string model_path = parser.get<std::string>("model");
    std::string palm_model_path = parser.get<std::string>("palm_model");
    int bt = parser.get<int>("backend_target");
    float conf_thresh = parser.get<float>("conf_threshold");
    bool save = parser.get<bool>("save");
    bool vis = parser.get<bool>("vis");

    auto [backend_id, target_id] = backend_target_pairs[bt];

    MPPalmDet palm_detector(palm_model_path, 0.3f, 0.6f, 5000, backend_id, target_id);
    MPHandPose hand_detector(model_path, conf_thresh, backend_id, target_id);

    if (!input_path.empty()) {
        cv::Mat image = cv::imread(input_path);
        if (image.empty()) {
            std::cerr << "Failed to read image from " << input_path << "\n";
            return -1;
        }

        std::vector<std::vector<float>> palms = palm_detector.infer(image);
        std::vector<HandPose> hands;

        for (const auto& palm : palms) {
            HandPose hand;
            if (hand_detector.infer(image, palm, hand)) {
                hands.push_back(hand);
            }
        }

        cv::Mat annotated = visualize(image, hands, true);

        if (palms.empty()) {
            std::cout << "No palm detected!\n";
        } else {
            std::cout << "Palm detected!\n";
        }

        if (save) {
            cv::imwrite("result.jpg", annotated);
            std::cout << "Results saved to result.jpg\n";
        }

        if (vis) {
            cv::namedWindow(input_path, cv::WINDOW_AUTOSIZE);
            cv::imshow(input_path, annotated);
            cv::waitKey(0);
        }
    } else {
        int device_id = 0;
        cv::VideoCapture cap(device_id);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera device " << device_id << "\n";
            return -1;
        }

        cv::TickMeter tm;
        cv::Mat frame;

        while (cv::waitKey(1) < 0) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << "No frames grabbed!\n";
                break;
            }

            std::vector<std::vector<float>> palms = palm_detector.infer(frame);
            std::vector<HandPose> hands;

            tm.start();
            for (const auto& palm : palms) {
                HandPose hand;
                if (hand_detector.infer(frame, palm, hand)) {
                    hands.push_back(hand);
                }
            }
            tm.stop();

            cv::Mat annotated = visualize(frame, hands);
            if (!palms.empty()) {
                cv::putText(annotated, cv::format("FPS: %.2f", tm.getFPS()),
                            cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
            }
            cv::imshow("MediaPipe Handpose Detection Demo", annotated);
            tm.reset();
        }
    }

    return 0;
}
