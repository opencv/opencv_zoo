#include <opencv2/opencv.hpp>
#include "opencv2/dnn.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
#include <map>
#include <string>
#include <numeric>

namespace fs = std::filesystem;

// YoutuReID class for person re-identification
class YoutuReID {
public:
    YoutuReID(const std::string& model_path,
              const cv::Size& input_size = cv::Size(128, 256),
              int output_dim = 768,
              const cv::Scalar& mean = cv::Scalar(0.485, 0.456, 0.406),
              const cv::Scalar& std = cv::Scalar(0.229, 0.224, 0.225),
              int backend_id = 0,
              int target_id = 0)
        : model_path_(model_path), input_size_(input_size),
          output_dim_(output_dim), mean_(mean), std_(std),
          backend_id_(backend_id), target_id_(target_id)
    {
        
        model_ = cv::dnn::readNet(model_path_);
        model_.setPreferableBackend(backend_id_);
        model_.setPreferableTarget(target_id_);
    }

    void setBackendAndTarget(int backend_id, int target_id) {
        backend_id_ = backend_id;
        target_id_ = target_id;
        model_.setPreferableBackend(backend_id_);
        model_.setPreferableTarget(target_id_);
    }

    void setInputSize(const cv::Size& input_size) {
        input_size_ = input_size;
    }

    // Preprocess image by resizing, normalizing, and creating a blob
    cv::Mat preprocess(const cv::Mat& image) {
        cv::Mat img;
        cv::cvtColor(image, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1.0 / 255.0);

        // Normalize each channel separately
        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);
        channels[0] = (channels[0] - mean_[0]) / std_[0];
        channels[1] = (channels[1] - mean_[1]) / std_[1];
        channels[2] = (channels[2] - mean_[2]) / std_[2];
        cv::merge(channels, img);

        return cv::dnn::blobFromImage(img);
    }

    // Run inference to extract feature vector
    cv::Mat infer(const cv::Mat& image) {
        cv::Mat input_blob = preprocess(image);
        model_.setInput(input_blob);
        cv::Mat features = model_.forward();

        if (features.dims == 4 && features.size[2] == 1 && features.size[3] == 1) {
            features = features.reshape(1, {1, features.size[1]});
        }

        return features;
    }

    // Perform query, comparing each query image to each gallery image
    std::vector<std::vector<int>> query(const std::vector<cv::Mat>& query_img_list,
                                        const std::vector<cv::Mat>& gallery_img_list,
                                        int topK = 5) {
        std::vector<cv::Mat> query_features_list, gallery_features_list;
        cv::Mat query_features, gallery_features;

        for (const auto& q_img : query_img_list) {
            cv::Mat feature = infer(q_img);
            query_features_list.push_back(feature.clone());
        }
        cv::vconcat(query_features_list, query_features);
        normalizeFeatures(query_features);

        for (const auto& g_img : gallery_img_list) {
            cv::Mat feature = infer(g_img);
            gallery_features_list.push_back(feature.clone());
        }
        cv::vconcat(gallery_features_list, gallery_features);
        normalizeFeatures(gallery_features);

        cv::Mat dist = query_features * gallery_features.t();
        return getTopK(dist, topK);
    }

private:
    // Normalize feature vectors row-wise to unit length
    void normalizeFeatures(cv::Mat& features) {
        const float epsilon = 1e-6;
        for (int i = 0; i < features.rows; ++i) {
            cv::Mat featureRow = features.row(i);
            float norm = cv::norm(featureRow, cv::NORM_L2);
            if (norm < epsilon) {
                norm = epsilon;
            }
            featureRow /= norm;
        }
    }

    // Retrieve Top-K indices from similarity matrix
    std::vector<std::vector<int>> getTopK(const cv::Mat& dist, int topK) {
        std::vector<std::vector<int>> indices(dist.rows);
        
        for (int i = 0; i < dist.rows; ++i) {
            std::vector<std::pair<float, int>> sim_index_pairs;
            for (int j = 0; j < dist.cols; ++j) {
                sim_index_pairs.emplace_back(dist.at<float>(i, j), j);
            }
            std::sort(sim_index_pairs.begin(), sim_index_pairs.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first;
                      });

            for (int k = 0; k < topK && k < sim_index_pairs.size(); ++k) {
                indices[i].push_back(sim_index_pairs[k].second);
            }
        }
        return indices;
    }

    std::string model_path_;
    cv::Size input_size_;
    int output_dim_;
    cv::Scalar mean_, std_;
    int backend_id_;
    int target_id_;
    cv::dnn::Net model_;
};

// Load images and file names from a directory, resizing them
std::pair<std::vector<cv::Mat>, std::vector<std::string>> readImagesFromDirectory(const std::string& img_dir, int w = 128, int h = 256) {
    std::vector<cv::Mat> img_list;
    std::vector<std::string> file_list;

    for (const auto& entry : fs::directory_iterator(img_dir)) {
        std::string file_name = entry.path().filename().string();
        cv::Mat img = cv::imread(entry.path().string());
        if (!img.empty()) {
            cv::resize(img, img, cv::Size(w, h));
            img_list.push_back(img);
            file_list.push_back(file_name);
        }
    }
    return {img_list, file_list};
}

// Visualize query and gallery results by creating concatenated images
std::map<std::string, cv::Mat> visualize(
    const std::map<std::string, std::vector<std::string>>& results,
    const std::string& query_dir, 
    const std::string& gallery_dir,
    const cv::Size& output_size = cv::Size(128, 384)) {

    std::map<std::string, cv::Mat> results_vis;

    for (const auto& [query_file, top_matches] : results) {
        cv::Mat query_img = cv::imread(query_dir + "/" + query_file);
        if (query_img.empty()) continue;

        cv::resize(query_img, query_img, output_size);
        cv::copyMakeBorder(query_img, query_img, 5, 5, 5, 5, 
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::putText(query_img, "Query", cv::Point(10, 30), 
                    cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::Mat concat_img = query_img;

        for (size_t i = 0; i < top_matches.size(); ++i) {
            cv::Mat gallery_img = cv::imread(gallery_dir + "/" + top_matches[i]);
            if (gallery_img.empty()) continue;

            cv::resize(gallery_img, gallery_img, output_size);
            cv::copyMakeBorder(gallery_img, gallery_img, 5, 5, 5, 5, 
                               cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
            cv::putText(gallery_img, "G" + std::to_string(i), cv::Point(10, 30), 
                        cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            cv::hconcat(concat_img, gallery_img, concat_img);
        }
        results_vis[query_file] = concat_img;
    }
    return results_vis;
}

void printHelpMessage() {
    std::cout << "usage: demo.cpp [-h] [--query_dir QUERY_DIR] [--gallery_dir GALLERY_DIR] "
              << "[--backend_target BACKEND_TARGET] [--topk TOPK] [--model MODEL] [--save] [--vis]\n\n"
              << "ReID baseline models from Tencent Youtu Lab\n\n"
              << "optional arguments:\n"
              << "  -h, --help            show this help message and exit\n"
              << "  --query_dir QUERY_DIR, -q QUERY_DIR\n"
              << "                        Query directory.\n"
              << "  --gallery_dir GALLERY_DIR, -g GALLERY_DIR\n"
              << "                        Gallery directory.\n"
              << "  --backend_target BACKEND_TARGET, -bt BACKEND_TARGET\n"
              << "                        Choose one of the backend-target pair to run this demo: 0: (default) OpenCV implementation + "
                 "CPU, 1: CUDA + GPU (CUDA), 2: CUDA + GPU (CUDA FP16), 3: TIM-VX + NPU, 4: CANN + NPU\n"
              << "  --topk TOPK           Top-K closest from gallery for each query.\n"
              << "  --model MODEL, -m MODEL\n"
              << "                        Path to the model.\n"
              << "  --save, -s            Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in "
                 "case of camera input.\n"
              << "  --vis, -v             Usage: Specify to open a new window to show results. Invalid in case of camera input.\n";
}

int main(int argc, char** argv) {
    // CommandLineParser setup
    cv::CommandLineParser parser(argc, argv,
        "{help h | | Show help message.}"
        "{query_dir q | | Query directory.}"
        "{gallery_dir g | | Gallery directory.}"
        "{backend_target bt | 0 | Choose one of the backend-target pair to run this demo: 0: (default) OpenCV implementation + CPU, "
        "1: CUDA + GPU (CUDA), 2: CUDA + GPU (CUDA FP16), 3: TIM-VX + NPU, 4: CANN + NPU}"
        "{topk k | 10 | Top-K closest from gallery for each query.}"
        "{model m | person_reid_youtu_2021nov.onnx | Path to the model.}"
        "{save s | false | Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.}"
        "{vis v | false | Usage: Specify to open a new window to show results. Invalid in case of camera input.}");

    if (parser.has("help")) {
        printHelpMessage();
        return 0;
    }

    std::string query_dir = parser.get<std::string>("query_dir");
    std::string gallery_dir = parser.get<std::string>("gallery_dir");
    int backend_target = parser.get<int>("backend_target");
    int topK = parser.get<int>("topk");
    std::string model_path = parser.get<std::string>("model");
    bool save_flag = parser.get<bool>("save");
    bool vis_flag = parser.get<bool>("vis");

    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

    const std::vector<std::pair<int, int>> backend_target_pairs = {
        {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
        {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16},
        {cv::dnn::DNN_BACKEND_TIMVX, cv::dnn::DNN_TARGET_NPU},
        {cv::dnn::DNN_BACKEND_CANN, cv::dnn::DNN_TARGET_NPU}
    };

    int backend_id = backend_target_pairs[backend_target].first;
    int target_id = backend_target_pairs[backend_target].second;

    YoutuReID reid(model_path, cv::Size(128, 256), 768, 
                   cv::Scalar(0.485, 0.456, 0.406), 
                   cv::Scalar(0.229, 0.224, 0.225), 
                   backend_id, target_id);

    auto [query_imgs, query_file_list] = readImagesFromDirectory(query_dir);
    auto [gallery_imgs, gallery_file_list] = readImagesFromDirectory(gallery_dir);

    auto indices = reid.query(query_imgs, gallery_imgs, topK);

    std::map<std::string, std::vector<std::string>> results;
    for (size_t i = 0; i < query_file_list.size(); ++i) {
        std::vector<std::string> top_matches;
        for (int idx : indices[i]) {
            top_matches.push_back(gallery_file_list[idx]);
        }
        results[query_file_list[i]] = top_matches;
        std::cout << "Query: " << query_file_list[i] << "\n";
        std::cout << "\tTop-" << topK << " from gallery: ";
        for (const auto& match : top_matches) {
            std::cout << match << " ";
        }
        std::cout << std::endl;
    }

    std::map<std::string, cv::Mat> results_vis = visualize(results, query_dir, gallery_dir);

    if (save_flag) {
        for (const auto& [query_file, result_img] : results_vis) {
            std::string save_path = "result-" + query_file;
            cv::imwrite(save_path, result_img);
        }
    }

    if (vis_flag) {
        for (const auto& [query_file, result_img] : results_vis) {
            cv::namedWindow("result-" + query_file, cv::WINDOW_AUTOSIZE);
            cv::imshow("result-" + query_file, result_img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    return 0;
}
