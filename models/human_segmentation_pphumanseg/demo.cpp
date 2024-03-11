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

class PPHS
{
private:
    Net model;
    string modelPath;
    
    Scalar imageMean = Scalar(0.5,0.5,0.5);
    Scalar imageStd = Scalar(0.5,0.5,0.5);
    Size modelInputSize = Size(192, 192);
    Size currentSize;

    const String inputNames = "x";
    const String outputNames = "save_infer_model/scale_0.tmp_1";

    int backend_id;
    int target_id;
    
public:
    PPHS(const string& modelPath, 
        int backend_id = 0, 
        int target_id = 0) 
      : modelPath(modelPath), backend_id(backend_id), target_id(target_id)
    {
        this->model = readNet(modelPath);
        this->model.setPreferableBackend(backend_id);
        this->model.setPreferableTarget(target_id);
    }

    Mat preprocess(const Mat image)
    {
        this->currentSize = image.size();
        Mat preprocessed = Mat::zeros(this->modelInputSize, image.type());
        resize(image, preprocessed, this->modelInputSize);

        // image normalization
        preprocessed.convertTo(preprocessed, CV_32F, 1.0 / 255.0);
        preprocessed -= imageMean;
        preprocessed /= imageStd;

        return blobFromImage(preprocessed);;
    }

    Mat infer(const Mat image)
    {
        Mat inputBlob = preprocess(image);

        this->model.setInput(inputBlob, this->inputNames);
        Mat outputBlob = this->model.forward(this->outputNames);

        return postprocess(outputBlob);
    }

    Mat postprocess(Mat image) 
    {
        reduceArgMax(image,image,1);
        image = image.reshape(1,image.size[2]);
        image.convertTo(image, CV_32F);
        resize(image, image, this->currentSize, 0, 0, INTER_LINEAR);
        image.convertTo(image, CV_8U);

        return image;
    }

};


vector<uint8_t> getColorMapList(int num_classes) {
    num_classes += 1;

    vector<uint8_t> cm(num_classes*3, 0);

    int lab, j;

    for (int i = 0; i < num_classes; ++i) {
        lab = i;
        j = 0;

        while(lab){
            cm[i] |= (((lab >> 0) & 1) << (7 - j));
            cm[i+num_classes] |= (((lab >> 1) & 1) << (7 - j));
            cm[i+2*num_classes] |= (((lab >> 2) & 1) << (7 - j));
            ++j;
            lab >>= 3; 
        }

    }

    cm.erase(cm.begin(), cm.begin()+3);

    return cm;
};

Mat visualize(const Mat& image, const Mat& result, float fps = -1.f, float weight = 0.4)
{
    const Scalar& text_color = Scalar(0, 255, 0);
    Mat output_image = image.clone();

    vector<uint8_t> color_map = getColorMapList(256);

    Mat cmm(color_map);

    cmm = cmm.reshape(1,{3,256});

    if (fps >= 0)
    {
        putText(output_image, format("FPS: %.2f", fps), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    Mat c1, c2, c3;

    LUT(result, cmm.row(0), c1);
    LUT(result, cmm.row(1), c2);
    LUT(result, cmm.row(2), c3);

    Mat pseudo_img;
    merge(std::vector<Mat>{c1,c2,c3}, pseudo_img);

    addWeighted(output_image, weight, pseudo_img, 1 - weight, 0, output_image);

    return output_image;
};

string keys =
"{ help  h          |                                                   | Print help message. }"
"{ model m          | human_segmentation_pphumanseg_2023mar.onnx        | Usage: Path to the model, defaults to human_segmentation_pphumanseg_2023mar.onnx }"
"{ input i          |                                                   | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ backend_target t | 0                                                 | Choose one of the backend-target pair to run this demo:\n"
                                                                        "0: (default) OpenCV implementation + CPU,\n"
                                                                        "1: CUDA + GPU (CUDA),\n"
                                                                        "2: CUDA + GPU (CUDA FP16),\n"
                                                                        "3: TIM-VX + NPU,\n"
                                                                        "4: CANN + NPU}"
"{ save s           | false                                             | Specify to save results.}"
"{ vis v            | true                                              | Specify to open a window for result visualization.}"
;


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
 
    parser.about("Human Segmentation");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    string modelPath = parser.get<string>("model");
    string inputPath = parser.get<string>("input");
    uint8_t backendTarget = parser.get<uint8_t>("backend_target");
    bool saveFlag = parser.get<bool>("save");
    bool visFlag = parser.get<bool>("vis");

    if (modelPath.empty())
        CV_Error(Error::StsError, "Model file " + modelPath + " not found");

    PPHS humanSegmentationModel(modelPath, backend_target_pairs[backendTarget].first, backend_target_pairs[backendTarget].second);

    VideoCapture cap;
    if (!inputPath.empty())
        cap.open(samples::findFile(inputPath));
    else
        cap.open(0);
    
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot opend video or file");    

    Mat frame;
    Mat result;
    static const std::string kWinName = "Human Segmentation Demo";
    TickMeter tm;

    while (waitKey(1) < 0)
    {
        cap >> frame;

        if (frame.empty())
        {
            if(inputPath.empty())
                cout << "Frame is empty" << endl;
            break;
        }

        tm.start();
        result = humanSegmentationModel.infer(frame);
        tm.stop();
        
        Mat res_frame = visualize(frame, result, tm.getFPS());

        if(visFlag || inputPath.empty())
        {
            imshow(kWinName, res_frame);
            if(!inputPath.empty())
                waitKey(0);
        }
        if(saveFlag)
        {
            cout << "Results are saved to result.jpg" << endl;

            imwrite("result.jpg", res_frame);
        }
    }
    
    return 0;
}   

