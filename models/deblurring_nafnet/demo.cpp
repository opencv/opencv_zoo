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

class Nafnet {
public:
    Nafnet(const string& modelPath) {
        loadModel(modelPath);
    }

    // Function to set up the input image and process it
    void process(const Mat& image, Mat& result) {
        Mat blob = blobFromImage(image, 0.00392, Size(image.cols, image.rows), Scalar(0, 0, 0), true, false, CV_32F);
        net.setInput(blob);
        Mat output = net.forward();
        postProcess(output, result);
    }

private:
    Net net;

    // Load Model
    void loadModel(const string modelPath) {
        net = readNetFromONNX(modelPath);
        net.setPreferableBackend(DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }

    void postProcess(const Mat& output, Mat& result) {
        Mat output_transposed(3, &output.size[1], CV_32F, const_cast<void*>(reinterpret_cast<const void*>(output.ptr<float>())));

        vector<Mat> channels;
        for (int i = 0; i < 3; ++i) {
            channels.push_back(Mat(output_transposed.size[1], output_transposed.size[2], CV_32F,
                                        output_transposed.ptr<float>(i)));
        }
        merge(channels, result);
        result.convertTo(result, CV_8UC3, 255.0);
        cvtColor(result, result, COLOR_RGB2BGR);
    }
};

int main(int argc, char** argv) {
    const string about =
        "This sample demonstrates deblurring with nafnet deblurring model.\n\n";
    const string keys =
        "{ help h          |                                         | Print help message. }"
        "{ input i         | example_outputs/licenseplate_motion.jpg | Path to input image.}"
        "{ model           |     deblurring_nafnet_2025may.onnx      | Path to the nafnet deblurring onnx model file }";

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

    Mat image = imread(parser.get<String>("input"));
    if (image.empty()) {
        cerr << "Error: Input image could not be loaded." << endl;
        return -1;
    }

    // Create an instance of Dexined
    Nafnet nafnet(model);

    Mat result;
    nafnet.process(image, result);

    imshow("Input", image);
    imshow("Output", result);
    waitKey(0);

    destroyAllWindows();
    return 0;
}
