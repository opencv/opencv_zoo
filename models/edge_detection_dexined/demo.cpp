#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Dexined {
public:
    Dexined(const string& modelPath) {
        loadModel(modelPath);
    }

    // Function to set up the input image and process it
    void processFrame(const Mat& image, Mat& result) {
        Mat blob = blobFromImage(image, 1.0, Size(512, 512), Scalar(103.5, 116.2, 123.6), false, false, CV_32F);
        net.setInput(blob);
        applyDexined(image, result);
    }

private:
    Net net;

    // Load Model
    void loadModel(const string modelPath) {
        net = readNetFromONNX(modelPath);
        net.setPreferableBackend(DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }

    // Function to apply sigmoid activation
    static void sigmoid(Mat& input) {
        exp(-input, input);          // e^-input
        input = 1.0 / (1.0 + input); // 1 / (1 + e^-input)
    }

    // Function to process the neural network output to generate edge maps
    static pair<Mat, Mat> postProcess(const vector<Mat>& output, int height, int width) {
        vector<Mat> preds;
        preds.reserve(output.size());
        for (const Mat &p : output) {
            Mat img;
            Mat processed;
            if (p.dims == 4 && p.size[0] == 1 && p.size[1] == 1) {
                processed = p.reshape(0, {p.size[2], p.size[3]});
            } else {
                processed = p.clone();
            }
            sigmoid(processed);
            normalize(processed, img, 0, 255, NORM_MINMAX, CV_8U);
            resize(img, img, Size(width, height));
            preds.push_back(img);
        }
        Mat fuse = preds.back();
        Mat ave = Mat::zeros(height, width, CV_32F);
        for (Mat &pred : preds) {
            Mat temp;
            pred.convertTo(temp, CV_32F);
            ave += temp;
        }
        ave /= static_cast<float>(preds.size());
        ave.convertTo(ave, CV_8U);
        return {fuse, ave};
    }

    // Function to apply the Dexined model
    void applyDexined(const Mat& image, Mat& result) {
        int originalWidth = image.cols;
        int originalHeight = image.rows;
        vector<Mat> outputs;
        net.forward(outputs);
        pair<Mat, Mat> res = postProcess(outputs, originalHeight, originalWidth);
        result = res.first; // or res.second for average edge map
    }
};

int main(int argc, char** argv) {
    const string about =
        "This sample demonstrates edge detection with dexined edge detection techniques.\n\n";
    const string keys =
        "{ help h          |                                     | Print help message. }"
        "{ input i         |                                     | Path to input image or video file. Skip this argument to capture frames from a camera.}"
        "{ model           | edge_detection_dexined_2024sep.onnx | Path to the dexined.onnx model file }";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return -1;
    }

    parser = CommandLineParser(argc, argv, keys);
    string model = parser.get<String>("model");
    parser.about(about);

    VideoCapture cap;
    if (parser.has("input"))
        cap.open(samples::findFile(parser.get<String>("input")));
    else
        cap.open(0);

    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    moveWindow("Output", 200, 0);

    // Create an instance of Dexined
    Dexined dexined(model);
    Mat image;

    for (;;){
        cap >> image;
        if (image.empty())
        {
            cout << "Press any key to exit" << endl;
            waitKey();
            break;
        }

        Mat result;
        dexined.processFrame(image, result);

        imshow("Input", image);
        imshow("Output", result);
        int key = waitKey(1);
        if (key == 27 || key == 'q')
        {
            break;
        }
    }
    destroyAllWindows();
    return 0;
}
