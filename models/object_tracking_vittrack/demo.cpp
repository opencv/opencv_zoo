#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

struct TrackingResult
{
    bool isLocated;
    Rect bbox;
    float score;
};

class VitTrack
{
public:

    VitTrack(const string& model_path, int backend_id = 0, int target_id = 0) 
    {
        params.net = model_path;
        params.backend = backend_id;
        params.target = target_id;
        model = TrackerVit::create(params);
    }

    void init(const Mat& image, const Rect& roi)
    {
        model->init(image, roi);
    }

    TrackingResult infer(const Mat& image)
    {
        TrackingResult result;
        result.isLocated = model->update(image, result.bbox);
        result.score = model->getTrackingScore();
        return result;
    }

private:
    TrackerVit::Params params;
    Ptr<TrackerVit> model;
};

Mat visualize(const Mat& image, const Rect& bbox, float score, bool isLocated, double fps = -1.0,
                  const Scalar& box_color = Scalar(0, 255, 0), const Scalar& text_color = Scalar(0, 255, 0),
                  double fontScale = 1.0, int fontSize = 1)
{
    Mat output = image.clone();
    int h = output.rows;
    int w = output.cols;

    if (fps >= 0)
    {
        putText(output, "FPS: " + to_string(fps), Point(0, 30), FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize);
    }

    if (isLocated && score >= 0.3)
    {
        rectangle(output, bbox, box_color, 2);
        putText(output, format("%.2f", score), Point(bbox.x, bbox.y + 25),
                    FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize);
    }
    else
    {
        Size text_size = getTextSize("Target lost!", FONT_HERSHEY_DUPLEX, fontScale, fontSize, nullptr);
        int text_x = (w - text_size.width) / 2;
        int text_y = (h - text_size.height) / 2;
        putText(output, "Target lost!", Point(text_x, text_y), FONT_HERSHEY_DUPLEX, fontScale, Scalar(0, 0, 255), fontSize);
    }

    return output;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
        "{help  h           |                                       | Print help message. }"
        "{input i           |                                       |Set path to the input video. Omit for using default camera.}"
        "{model_path        |object_tracking_vittrack_2023sep.onnx  |Set model path}"
        "{backend_target bt |0                                      |Choose backend-target pair: 0 - OpenCV implementation + CPU, 1 - CUDA + GPU (CUDA), 2 - CUDA + GPU (CUDA FP16), 3 - TIM-VX + NPU, 4 - CANN + NPU}"
        "{save s            |false                                  |Specify to save a file with results.}"
        "{vis v             |true                                   |Specify to open a new window to show results.}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string input = parser.get<string>("input");
    string model_path = parser.get<string>("model_path");
    int backend_target = parser.get<int>("backend_target");
    bool save = parser.get<bool>("save");
    bool vis = parser.get<bool>("vis");

    vector<vector<int>> backend_target_pairs =
    {
        {DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
        {DNN_BACKEND_CUDA, DNN_TARGET_CUDA},
        {DNN_BACKEND_CUDA, DNN_TARGET_CUDA_FP16},
        {DNN_BACKEND_TIMVX, DNN_TARGET_NPU},
        {DNN_BACKEND_CANN, DNN_TARGET_NPU}
    };

    int backend_id = backend_target_pairs[backend_target][0];
    int target_id = backend_target_pairs[backend_target][1];

    // Create VitTrack tracker
    VitTrack tracker(model_path, backend_id, target_id);

    // Open video capture
    VideoCapture video;
    if (input.empty())
    {
        video.open(0);  // Default camera
    }
    else
    {
        video.open(input);
    }

    if (!video.isOpened())
    {
        cerr << "Error: Could not open video source" << endl;
        return -1;
    }

    // Select an object
    Mat first_frame;
    video >> first_frame;

    if (first_frame.empty())
    {
        cerr << "No frames grabbed!" << endl;
        return -1;
    }

    Mat first_frame_copy = first_frame.clone();
    putText(first_frame_copy, "1. Drag a bounding box to track.", Point(0, 25), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0));
    putText(first_frame_copy, "2. Press ENTER to confirm", Point(0, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0));
    Rect roi = selectROI("VitTrack Demo", first_frame_copy);

    if (roi.area() == 0)
    {
        cerr << "No ROI is selected! Exiting..." << endl;
        return -1;
    }
    else
    {
        cout << "Selected ROI: " << roi << endl;
    }

    // Create VideoWriter if save option is specified
    VideoWriter output_video;
    if (save)
    {
        Size frame_size = first_frame.size();
        output_video.open("output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), video.get(CAP_PROP_FPS), frame_size);
        if (!output_video.isOpened())
        {
            cerr << "Error: Could not create output video stream" << endl;
            return -1;
        }
    }

    // Initialize tracker with ROI
    tracker.init(first_frame, roi);

    // Track frame by frame
    TickMeter tm;
    while (waitKey(1) < 0)
    {
        video >> first_frame;
        if (first_frame.empty())
        {
            cout << "End of video" << endl;
            break;
        }

        // Inference
        tm.start();
        TrackingResult result = tracker.infer(first_frame);
        tm.stop();

        // Visualize
        Mat frame = first_frame.clone();
        frame = visualize(frame, result.bbox, result.score, result.isLocated, tm.getFPS());

        if (save)
        {
            output_video.write(frame);
        }

        if (vis)
        {
            imshow("VitTrack Demo", frame);
        }
        tm.reset();
    }

    if (save)
    {
        output_video.release();
    }

    video.release();
    destroyAllWindows();

    return 0;
}
