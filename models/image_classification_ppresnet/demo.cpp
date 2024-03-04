#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <algorithm>
#include <fstream>

class PPResNet {
public:
    PPResNet(const std::string& modelPath, int topK, int backendId, int targetId)
        : _topK(topK) {
        _model = cv::dnn::readNet(modelPath);
        _model.setPreferableBackend(backendId);
        _model.setPreferableTarget(targetId);
        loadLabels();
    }

    cv::Mat preprocess(const cv::Mat& image) {
        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32F, 1.0 / 255.0);
        cv::subtract(floatImage, _mean, floatImage);
        cv::divide(floatImage, _std, floatImage);
        return cv::dnn::blobFromImage(floatImage);
    }

    std::vector<std::string> infer(const cv::Mat& image) {
        assert(image.rows == _inputSize.height && image.cols == _inputSize.width);
        cv::Mat inputBlob = preprocess(image);
        _model.setInput(inputBlob, _inputName);
        cv::Mat outputBlob = _model.forward(_outputName);
        std::vector<std::string> results = postprocess(outputBlob);
        return results;
    }

    std::vector<std::string> postprocess(const cv::Mat& outputBlob) {
        std::vector<int> class_id_list;
        cv::sortIdx(outputBlob, class_id_list, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
        class_id_list.resize(std::min(_topK, static_cast<int>(outputBlob.cols)));
        std::vector<std::string> predicted_labels;
        for (int class_id : class_id_list) {
            predicted_labels.push_back(_labels[class_id]);
        }
        return predicted_labels;
    }

    void loadLabels() {
        std::ifstream labelsFile("labels.txt");
        if (labelsFile.is_open()) {
            std::string line;
            while (std::getline(labelsFile, line)) {
                _labels.push_back(line);
            }
            labelsFile.close();
        } else {
            std::cerr << "Unable to open labels file!" << std::endl;
        }
    }

private:
    cv::dnn::Net _model;
    int _topK;
    std::vector<std::string> _labels;
    const cv::Size _inputSize = cv::Size(224, 224);
    const cv::Scalar _mean = cv::Scalar(0.485, 0.456, 0.406);
    const cv::Scalar _std = cv::Scalar(0.229, 0.224, 0.225);
    std::string _inputName = "";
    std::string _outputName = "save_infer_model/scale_0.tmp_0";
};

const std::vector<std::vector<int>> backend_target_pairs = {
    {cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU},
    {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA},
    {cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA_FP16},
    {cv::dnn::DNN_BACKEND_TIMVX, cv::dnn::DNN_TARGET_NPU},
    {cv::dnn::DNN_BACKEND_CANN, cv::dnn::DNN_TARGET_NPU}
};

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv,
        "{ input i               |                                               | Set input path to a certain image, omit if using camera.}"
        "{ model m               | image_classification_ppresnet50_2022jan.onnx  | Set model path.}"
        "{ top_k k               | 1                                             | Get top k predictions.}"
        "{ backend_target bt     | 0                                             | Choose one of computation backends: "
        "0: (default) OpenCV implementation + CPU, "
        "1: CUDA + GPU (CUDA), "
        "2: CUDA + GPU (CUDA FP16), "
        "3: TIM-VX + NPU, "
        "4: CANN + NPU}");

    std::string inputPath = parser.get<std::string>("input");
    std::string modelPath = parser.get<std::string>("model");
    int backendTarget = parser.get<int>("backend_target");
    int topK = parser.get<int>("top_k");

    int backendId = backend_target_pairs[backendTarget][0];
    int targetId = backend_target_pairs[backendTarget][1];

    PPResNet model(modelPath, topK, backendId, targetId);

    // Read image and get a 224x224 crop from a 256x256 resized
    cv::Mat image = cv::imread(inputPath);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(256, 256));
    image = image(cv::Rect(16, 16, 224, 224));

    // Inference
    auto predictions = model.infer(image);

    // Print result
    if (topK == 1)
    {
        std::cout << "Predicted Label: " << predictions[0] << std::endl;
    }
    else 
    {
        std::cout << "Predicted Top-K Labels (in decreasing confidence): " << std::endl;
        for (size_t i = 0; i < predictions.size(); ++i) 
        {
            std::cout << "(" << i+1 << ") " << predictions[i] << std::endl;
        }
    }

    return 0;
}
