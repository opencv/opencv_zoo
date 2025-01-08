/*
This sample inpaints the masked area in the given image.

Copyright (C) 20245, Bigvision LLC.
*/

#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class Lama {
public:
    Lama(const string& modelPath) {
        loadModel(modelPath);
    }

    // Function to set up the input image and process it
    void process(const Mat& image, const Mat& mask, Mat& result) {
        double aspectRatio = static_cast<double>(image.rows) / static_cast<double>(image.cols);

        Mat image_blob = blobFromImage(image, 1.0/255.0, Size(512, 512), Scalar(0, 0, 0), false, false, CV_32F);
        Mat mask_blob = blobFromImage(mask, 1.0, Size(512, 512), Scalar(0), false, false);

        mask_blob = (mask_blob > 0);
        mask_blob.convertTo(mask_blob, CV_32F);
        mask_blob = mask_blob/255.0;

        net.setInput(image_blob, "image");
        net.setInput(mask_blob, "mask");

        Mat output = net.forward();

        postProcess(output, result, aspectRatio);
    }
private:
    Net net;

    // Load Model
    void loadModel(const string modelPath) {
        net = readNetFromONNX(modelPath);
        net.setPreferableBackend(DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }

    void postProcess(const Mat& output, Mat& result, double aspectRatio) {
        Mat output_transposed(3, &output.size[1], CV_32F, const_cast<void*>(reinterpret_cast<const void*>(output.ptr<float>())));

        vector<Mat> channels;
        for (int i = 0; i < 3; ++i) {
            channels.push_back(Mat(output_transposed.size[1], output_transposed.size[2], CV_32F,
                                        output_transposed.ptr<float>(i)));
        }
        merge(channels, result);
        result.convertTo(result, CV_8U);

        int h = static_cast<int>(512 * aspectRatio);
        resize(result, result, Size(512, h));
    }
};


const string about = "This sample demonstrates image inpainting with lama inpainting technique.\n\n";

const string keys =
    "{help    h  |                              | show help message}"
    "{input   i  |                              | Path to input image}"
    "{ model     | inpainting_lama_2024jan.onnx | Path to the lama onnx model file }";

bool drawing = false;
Mat maskGray;
int brush_size = 25;

static void drawMask(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        drawing = true;
    } else if (event == EVENT_MOUSEMOVE) {
        if (drawing) {
            circle(maskGray, Point(x, y), brush_size, Scalar(255), -1);
        }
    } else if (event == EVENT_LBUTTONUP) {
        drawing = false;
    }
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        cout<<about<<endl;
        parser.printMessage();
        return 0;
    }
    parser = CommandLineParser(argc, argv, keys);
    parser.about(about);

    const string model = parser.get<String>("model");

    int height = 512;
    int width = 512;
    int stdSize = 20;
    int stdWeight = 400;
    int stdImgSize = 512;
    int imgWidth = -1; // Initialization
    int fontSize = 50;
    int fontWeight = 500;

    FontFace fontFace("sans");
    Lama lama(model);

    Mat image = imread(parser.get<String>("input"));
    if (image.empty()) {
        cerr << "Error: Input image could not be loaded." << endl;
        return -1;
    }

    imgWidth = min(image.rows, image.cols);
    fontSize = min(fontSize, (stdSize*imgWidth)/stdImgSize);
    fontWeight = min(fontWeight, (stdWeight*imgWidth)/stdImgSize);

    maskGray = Mat::zeros(image.size(), CV_8U);

    namedWindow("Draw Mask");
    setMouseCallback("Draw Mask", drawMask);

    const string label = "Draw the mask on the image. Press space bar when done ";

    for(;;) {
        Mat displayImage = image.clone();
        Mat overlay = image.clone();

        double alpha = 0.5;
        Rect r = getTextSize(Size(), label, Point(), fontFace, fontSize, fontWeight);
        r.height += 2 * fontSize; // padding
        r.width += 10; // padding
        rectangle(overlay, r, Scalar::all(255), FILLED);
        addWeighted(overlay, alpha, displayImage, 1 - alpha, 0, displayImage);
        putText(displayImage, label, Point(10, fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);
        putText(displayImage, "Press 'i' to increase and 'd' to decrease brush size", Point(10, 2*fontSize), Scalar(0,0,0), fontFace, fontSize, fontWeight);

        displayImage.setTo(Scalar(255, 255, 255), maskGray > 0); // Highlight mask area
        imshow("Draw Mask", displayImage);

        char key = waitKey(1);
        if (key == 'i') {
            brush_size += 1;
            cout << "Brush size increased to " << brush_size << endl;
        } else if (key == 'd') {
            brush_size = max(1, brush_size - 1);
            cout << "Brush size decreased to " << brush_size << endl;
        } else if (key == ' ') {
            break;
        } else if (key == 27){
            return -1;
        }
    }
    destroyAllWindows();

    Mat result;
    lama.process(image, maskGray, result);

    imshow("Inpainted Output", result);
    waitKey(0);

    return 0;
}
