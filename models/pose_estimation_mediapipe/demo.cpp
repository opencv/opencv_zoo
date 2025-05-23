#include <vector>
#include <string>
#include <utility>
#include <cmath>

#include <opencv2/opencv.hpp>

const long double _M_PI = 3.141592653589793238L;
using namespace std;
using namespace cv;
using namespace dnn;

vector< pair<dnn::Backend, dnn::Target> > backendTargetPairs = {
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA_FP16),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_TIMVX, dnn::DNN_TARGET_NPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CANN, dnn::DNN_TARGET_NPU) };


Mat getMediapipeAnchor();

class MPPersonDet {
private:
    Net net;
    string modelPath;
    Size inputSize;
    float scoreThreshold;
    float nmsThreshold;
    dnn::Backend backendId;
    dnn::Target targetId;
    int topK;
    Mat anchors;

public:
    MPPersonDet(string modPath, float nmsThresh = 0.3, float scoreThresh = 0.5, int tok=5000 , dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU) :
        modelPath(modPath), nmsThreshold(nmsThresh),
        scoreThreshold(scoreThresh), topK(tok),
        backendId(bId), targetId(tId)
    {
        this->inputSize = Size(224, 224);
        this->net = readNet(this->modelPath);
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
        this->anchors = getMediapipeAnchor();
    }

    pair<Mat, Size> preprocess(Mat img)
    {
        Mat blob;
        Image2BlobParams paramMediapipe;
        paramMediapipe.datalayout = DNN_LAYOUT_NCHW;
        paramMediapipe.ddepth = CV_32F;
        paramMediapipe.mean = Scalar::all(127.5);
        paramMediapipe.scalefactor = Scalar::all(1/127.5);
        paramMediapipe.size = this->inputSize;
        paramMediapipe.swapRB = true;
        paramMediapipe.paddingmode = DNN_PMODE_LETTERBOX;

        double ratio = min(this->inputSize.height / double(img.rows), this->inputSize.width / double(img.cols));
        Size padBias(0, 0);
        if (img.rows != this->inputSize.height || img.cols != this->inputSize.width)
        {
            // keep aspect ratio when resize
            Size ratioSize(int(img.cols * ratio), int(img.rows* ratio));
            int padH = this->inputSize.height - ratioSize.height;
            int padW = this->inputSize.width - ratioSize.width;
            padBias.width = padW / 2;
            padBias.height = padH / 2;
        }
        blob = blobFromImageWithParams(img, paramMediapipe);
        padBias = Size(int(padBias.width / ratio), int(padBias.height / ratio));
        return pair<Mat, Size>(blob, padBias);
    }

   Mat infer(Mat srcimg)
    {
        pair<Mat, Size> w = this->preprocess(srcimg);
        Mat inputBlob = get<0>(w);
        Size padBias = get<1>(w);
        this->net.setInput(inputBlob);
        vector<Mat> outs;
        this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
        Mat predictions = this->postprocess(outs, Size(srcimg.cols, srcimg.rows), padBias);
        return predictions;
    }

    Mat postprocess(vector<Mat> outputs, Size orgSize, Size padBias)
    {
        Mat score = outputs[1].reshape(0, outputs[1].size[0]);
        Mat boxLandDelta = outputs[0].reshape(outputs[0].size[0], outputs[0].size[1]);
        Mat boxDelta = boxLandDelta.colRange(0, 4);
        Mat landmarkDelta = boxLandDelta.colRange(4, boxLandDelta.cols);
        float scale = float(max(orgSize.height, orgSize.width));
        Mat mask = score < -100;
        score.setTo(-100, mask);
        mask = score > 100;
        score.setTo(100, mask);
        Mat deno;
        exp(-score, deno);
        divide(1.0, 1+deno, score);
        boxDelta.colRange(0, 1) = boxDelta.colRange(0, 1) / this->inputSize.width;
        boxDelta.colRange(1, 2) = boxDelta.colRange(1, 2) / this->inputSize.height;
        boxDelta.colRange(2, 3) = boxDelta.colRange(2, 3) / this->inputSize.width;
        boxDelta.colRange(3, 4) = boxDelta.colRange(3, 4) / this->inputSize.height;
        Mat xy1 = (boxDelta.colRange(0, 2) - boxDelta.colRange(2, 4) / 2 + this->anchors) * scale;
        Mat xy2 = (boxDelta.colRange(0, 2) + boxDelta.colRange(2, 4) / 2 + this->anchors) * scale;
        Mat boxes;
        hconcat(xy1, xy2, boxes);
        vector< Rect2d > rotBoxes(boxes.rows);
        boxes.colRange(0, 1) = boxes.colRange(0, 1) - padBias.width;
        boxes.colRange(1, 2) = boxes.colRange(1, 2) - padBias.height;
        boxes.colRange(2, 3) = boxes.colRange(2, 3) - padBias.width;
        boxes.colRange(3, 4) = boxes.colRange(3, 4) - padBias.height;
        for (int i = 0; i < boxes.rows; i++)
        {
            rotBoxes[i] = Rect2d(Point2d(boxes.at<float>(i, 0), boxes.at<float>(i, 1)), Point2d(boxes.at<float>(i, 2), boxes.at<float>(i, 3)));
        }
        vector<int> keep;
        NMSBoxes(rotBoxes, score, this->scoreThreshold, this->nmsThreshold, keep, 1.0f, this->topK);
        if (keep.size() == 0)
            return Mat();
        int nbCols = landmarkDelta.cols + boxes.cols + 1;
        Mat candidates(int(keep.size()), nbCols, CV_32FC1);
        int row = 0;
        for (auto idx : keep)
        {
            candidates.at<float>(row, nbCols - 1) = score.at<float>(idx);
            boxes.row(idx).copyTo(candidates.row(row).colRange(0, 4));
            candidates.at<float>(row, 4) = (landmarkDelta.at<float>(idx, 0) / this->inputSize.width + this->anchors.at<float>(idx,0)) * scale - padBias.width;
            candidates.at<float>(row, 5) = (landmarkDelta.at<float>(idx, 1) / this->inputSize.height + this->anchors.at<float>(idx, 1))* scale - padBias.height;
            candidates.at<float>(row, 6) = (landmarkDelta.at<float>(idx, 2) / this->inputSize.width + this->anchors.at<float>(idx, 0))* scale - padBias.width;
            candidates.at<float>(row, 7) = (landmarkDelta.at<float>(idx, 3) / this->inputSize.height + this->anchors.at<float>(idx, 1))* scale - padBias.height;
            candidates.at<float>(row, 8) = (landmarkDelta.at<float>(idx, 4) / this->inputSize.width + this->anchors.at<float>(idx, 0))* scale - padBias.width;
            candidates.at<float>(row, 9) = (landmarkDelta.at<float>(idx, 5) / this->inputSize.height + this->anchors.at<float>(idx, 1))* scale - padBias.height;
            candidates.at<float>(row, 10) = (landmarkDelta.at<float>(idx, 6) / this->inputSize.width + this->anchors.at<float>(idx, 0))* scale - padBias.width;
            candidates.at<float>(row, 11) = (landmarkDelta.at<float>(idx, 7) / this->inputSize.height + this->anchors.at<float>(idx, 1))* scale - padBias.height;
            row++;
        }
        return candidates;
       
    }


};

class MPPose {
private:
    Net net;
    string modelPath;
    Size inputSize;
    float confThreshold;
    dnn::Backend backendId;
    dnn::Target targetId;
    float personBoxPreEnlargeFactor;
    float personBoxEnlargeFactor;
    Mat anchors;

public:
    MPPose(string modPath, float confThresh = 0.5, dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU) :
        modelPath(modPath), confThreshold(confThresh),
        backendId(bId), targetId(tId)
    {
        this->inputSize = Size(256, 256);
        this->net = readNet(this->modelPath);
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
        this->anchors = getMediapipeAnchor();
        // RoI will be larger so the performance will be better, but preprocess will be slower.Default to 1.
        this->personBoxPreEnlargeFactor = 1;
        this->personBoxEnlargeFactor = 1.25;
    }

    tuple<Mat, Mat, float, Mat, Size> preprocess(Mat image, Mat person)
    {
        /***
                Rotate input for inference.
                Parameters:
                  image - input image of BGR channel order
                  face_bbox - human face bounding box found in image of format [[x1, y1], [x2, y2]] (top-left and bottom-right points)
                  person_landmarks - 4 landmarks (2 full body points, 2 upper body points) of shape [4, 2]
                Returns:
                  rotated_person - rotated person image for inference
                  rotate_person_bbox - person box of interest range
                  angle - rotate angle for person
                  rotation_matrix - matrix for rotation and de-rotation
                  pad_bias - pad pixels of interest range
        */
        //  crop and pad image to interest range
        Size padBias(0, 0); // left, top
        Mat personKeypoints = person.colRange(4, 12).reshape(0, 4);
        Point2f midHipPoint = Point2f(personKeypoints.row(0));
        Point2f fullBodyPoint = Point2f(personKeypoints.row(1));
        // # get RoI
        double fullDist = norm(midHipPoint - fullBodyPoint);
        Mat fullBoxf,fullBox;
        Mat v1 = Mat(midHipPoint) - fullDist, v2 = Mat(midHipPoint);
        vector<Mat> vmat = { Mat(midHipPoint) - fullDist, Mat(midHipPoint) + fullDist };
        hconcat(vmat, fullBoxf);
        // enlarge to make sure full body can be cover
        Mat cBox, centerBox, whBox;
        reduce(fullBoxf, centerBox, 1, REDUCE_AVG, CV_32F);
        whBox = fullBoxf.col(1) - fullBoxf.col(0);
        Mat newHalfSize = whBox * this->personBoxPreEnlargeFactor / 2;
        vmat[0] = centerBox - newHalfSize;
        vmat[1] = centerBox + newHalfSize;
        hconcat(vmat, fullBox);
        Mat personBox;
        fullBox.convertTo(personBox, CV_32S);
        // refine person bbox
        Mat idx = personBox.row(0) < 0;
        personBox.row(0).setTo(0, idx);
        idx = personBox.row(0) >= image.cols;
        personBox.row(0).setTo(image.cols , idx);
        idx = personBox.row(1) < 0;
        personBox.row(1).setTo(0, idx);
        idx = personBox.row(1) >= image.rows;
        personBox.row(1).setTo(image.rows, idx);        // crop to the size of interest

        image = image(Rect(personBox.at<int>(0, 0), personBox.at<int>(1, 0), personBox.at<int>(0, 1) - personBox.at<int>(0, 0), personBox.at<int>(1, 1) - personBox.at<int>(1, 0)));
        // pad to square
        int top = int(personBox.at<int>(1, 0) - fullBox.at<float>(1, 0));
        int left = int(personBox.at<int>(0, 0) - fullBox.at<float>(0, 0));
        int bottom = int(fullBox.at<float>(1, 1) - personBox.at<int>(1, 1));
        int right = int(fullBox.at<float>(0, 1) - personBox.at<int>(0, 1));
        copyMakeBorder(image, image, top, bottom, left, right, BORDER_CONSTANT, Scalar(0, 0, 0));
        padBias = Point(padBias) + Point(personBox.col(0)) - Point(left, top);
        // compute rotation
        midHipPoint -= Point2f(padBias);
        fullBodyPoint -= Point2f(padBias);
        float radians = float(_M_PI / 2 - atan2(-(fullBodyPoint.y - midHipPoint.y), fullBodyPoint.x - midHipPoint.x));
        radians = radians - 2 * float(_M_PI) * int((radians + _M_PI) / (2 * _M_PI));
        float angle = (radians * 180 / float(_M_PI));
        //  get rotation matrix*
        Mat rotationMatrix = getRotationMatrix2D(midHipPoint, angle, 1.0);
        //  get rotated image
        Mat rotatedImage;
        warpAffine(image, rotatedImage, rotationMatrix, Size(image.cols, image.rows));
        //  get landmark bounding box
        Mat blob;
        Image2BlobParams paramPoseMediapipe;
        paramPoseMediapipe.datalayout = DNN_LAYOUT_NHWC;
        paramPoseMediapipe.ddepth = CV_32F;
        paramPoseMediapipe.mean = Scalar::all(0);
        paramPoseMediapipe.scalefactor = Scalar::all(1 / 255.);
        paramPoseMediapipe.size = this->inputSize;
        paramPoseMediapipe.swapRB = true;
        paramPoseMediapipe.paddingmode = DNN_PMODE_NULL;
        blob = blobFromImageWithParams(rotatedImage, paramPoseMediapipe); // resize INTER_AREA becomes INTER_LINEAR in blobFromImage
        Mat rotatedPersonBox = (Mat_<float>(2, 2) << 0, 0, image.cols, image.rows);

        return tuple<Mat, Mat, float, Mat, Size>(blob, rotatedPersonBox, angle, rotationMatrix, padBias);
    }

    tuple<Mat, Mat, Mat, Mat, Mat, float> infer(Mat image, Mat person)
    {
        int h = image.rows;
        int w = image.cols;
        // Preprocess
        tuple<Mat, Mat, float, Mat, Size> tw;
        tw = this->preprocess(image, person);
        Mat inputBlob = get<0>(tw);
        Mat rotatedPersonBbox = get<1>(tw);
        float  angle = get<2>(tw);
        Mat rotationMatrix = get<3>(tw);
        Size padBias = get<4>(tw);

        // Forward
        this->net.setInput(inputBlob);
        vector<Mat> outputBlob;
        this->net.forward(outputBlob, this->net.getUnconnectedOutLayersNames());

        // Postprocess
        tuple<Mat, Mat, Mat, Mat, Mat, float> results;
        results = this->postprocess(outputBlob, rotatedPersonBbox, angle, rotationMatrix, padBias, Size(w, h));
        return results;// # [bbox_coords, landmarks_coords, conf]
    }

    tuple<Mat, Mat, Mat, Mat, Mat, float> postprocess(vector<Mat> blob, Mat rotatedPersonBox, float angle, Mat rotationMatrix, Size padBias, Size imgSize)
    {
        float valConf = blob[1].at<float>(0);
        if (valConf < this->confThreshold)
            return tuple<Mat, Mat, Mat, Mat, Mat, float>(Mat(), Mat(), Mat(), Mat(), Mat(), valConf);
        Mat landmarks = blob[0].reshape(0, 39);
        Mat mask = blob[2];
        Mat heatmap = blob[3];
        Mat landmarksWorld = blob[4].reshape(0, 39);

        Mat deno;
        // recover sigmoid score
        exp(-landmarks.colRange(3, landmarks.cols), deno);
        divide(1.0, 1 + deno, landmarks.colRange(3, landmarks.cols));
        // TODO: refine landmarks with heatmap. reference: https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/blazepose_tfjs/detector.ts#L577-L582
        heatmap = heatmap.reshape(0, heatmap.size[0]);
        // transform coords back to the input coords
        Mat whRotatedPersonPbox = rotatedPersonBox.row(1) - rotatedPersonBox.row(0);
        Mat scaleFactor = whRotatedPersonPbox.clone();
        scaleFactor.col(0) /= this->inputSize.width;
        scaleFactor.col(1) /= this->inputSize.height;
        landmarks.col(0) = (landmarks.col(0) - this->inputSize.width / 2) * scaleFactor.at<float>(0);
        landmarks.col(1) = (landmarks.col(1) - this->inputSize.height / 2) * scaleFactor.at<float>(1);
        landmarks.col(2) = landmarks.col(2) * max(scaleFactor.at<float>(1), scaleFactor.at<float>(0));
        Mat coordsRotationMatrix;
        getRotationMatrix2D(Point(0, 0), angle, 1.0).convertTo(coordsRotationMatrix, CV_32F);
        Mat rotatedLandmarks = landmarks.colRange(0, 2) * coordsRotationMatrix.colRange(0, 2);
        hconcat(rotatedLandmarks, landmarks.colRange(2, landmarks.cols), rotatedLandmarks);
        Mat rotatedLandmarksWorld = landmarksWorld.colRange(0, 2) * coordsRotationMatrix.colRange(0, 2);
        hconcat(rotatedLandmarksWorld, landmarksWorld.col(2), rotatedLandmarksWorld);
        // invert rotation
        Mat rotationComponent  = (Mat_<double>(2, 2) <<rotationMatrix.at<double>(0,0), rotationMatrix.at<double>(1, 0), rotationMatrix.at<double>(0, 1), rotationMatrix.at<double>(1, 1));
        Mat translationComponent = rotationMatrix(Rect(2, 0, 1, 2)).clone();
        Mat invertedTranslation = -rotationComponent * translationComponent;
        Mat inverseRotationMatrix;
        hconcat(rotationComponent, invertedTranslation, inverseRotationMatrix);
        Mat center, rc;
        reduce(rotatedPersonBox, rc, 0, REDUCE_AVG, CV_64F);
        hconcat(rc, Mat(1, 1, CV_64FC1, 1) , center);
        //  get box center
        Mat originalCenter(2, 1, CV_64FC1);
        originalCenter.at<double>(0) = center.dot(inverseRotationMatrix.row(0));
        originalCenter.at<double>(1) = center.dot(inverseRotationMatrix.row(1));
        for (int idxRow = 0; idxRow < rotatedLandmarks.rows; idxRow++)
        {
            landmarks.at<float>(idxRow, 0) = float(rotatedLandmarks.at<float>(idxRow, 0) + originalCenter.at<double>(0) + padBias.width); // 
            landmarks.at<float>(idxRow, 1) = float(rotatedLandmarks.at<float>(idxRow, 1) + originalCenter.at<double>(1) + padBias.height); // 
        }
        // get bounding box from rotated_landmarks
        double vmin0, vmin1, vmax0, vmax1;
        minMaxLoc(landmarks.col(0), &vmin0, &vmax0);
        minMaxLoc(landmarks.col(1), &vmin1, &vmax1);
        Mat bbox = (Mat_<float>(2, 2) << vmin0, vmin1, vmax0, vmax1);
        Mat centerBox;
        reduce(bbox, centerBox, 0, REDUCE_AVG, CV_32F);
        Mat whBox = bbox.row(1) - bbox.row(0);
        Mat newHalfSize = whBox * this->personBoxEnlargeFactor / 2;
        vector<Mat> vmat(2);
        vmat[0] = centerBox - newHalfSize;
        vmat[1] = centerBox + newHalfSize;
        vconcat(vmat, bbox);
        // invert rotation for mask
        mask = mask.reshape(1, 256);
        Mat invertRotationMatrix = getRotationMatrix2D(Point(mask.cols / 2, mask.rows / 2), -angle, 1.0);
        Mat invertRotationMask;
        warpAffine(mask, invertRotationMask, invertRotationMatrix, Size(mask.cols, mask.rows));
        // enlarge mask
        resize(invertRotationMask, invertRotationMask, Size(int(whRotatedPersonPbox.at<float>(0)), int(whRotatedPersonPbox.at<float>(1))));
        // crop and pad mask
        int minW = -min(padBias.width, 0);
        int minH= -min(padBias.height, 0);
        int left = max(padBias.width, 0);
        int top = max(padBias.height, 0);
        Size padOver = imgSize - Size(invertRotationMask.cols, invertRotationMask.rows) - padBias;
        int maxW = min(padOver.width, 0) + invertRotationMask.cols;
        int maxH = min(padOver.height, 0) + invertRotationMask.rows;
        int right = max(padOver.width, 0);
        int bottom = max(padOver.height, 0);
        invertRotationMask = invertRotationMask(Rect(minW, minH, maxW - minW, maxH - minH)).clone();
        copyMakeBorder(invertRotationMask, invertRotationMask, top, bottom, left, right, BORDER_CONSTANT, Scalar::all(0));
        // binarize mask
        threshold(invertRotationMask, invertRotationMask, 1, 255, THRESH_BINARY);

        /* 2*2 person bbox: [[x1, y1], [x2, y2]]
        # 39*5 screen landmarks: 33 keypoints and 6 auxiliary points with [x, y, z, visibility, presence], z value is relative to HIP
        # Visibility is probability that a keypoint is located within the frame and not occluded by another bigger body part or another object
        # Presence is probability that a keypoint is located within the frame
        # 39*3 world landmarks: 33 keypoints and 6 auxiliary points with [x, y, z] 3D metric x, y, z coordinate
        # img_height*img_width mask: gray mask, where 255 indicates the full body of a person and 0 means background
        # 64*64*39 heatmap: currently only used for refining landmarks, requires sigmod processing before use
        # conf: confidence of prediction*/
        return tuple<Mat , Mat, Mat, Mat, Mat, float>(bbox, landmarks, rotatedLandmarksWorld, invertRotationMask, heatmap, valConf);
    }
};

std::string keys =
"{ help  h          |                                               | Print help message. }"
"{ model m          | pose_estimation_mediapipe_2023mar.onnx        | Usage: Path to the model, defaults to person_detection_mediapipe_2023mar.onnx  }"
"{ input i          |                                               | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ conf_threshold   | 0.5                                           | Usage: Filter out hands of confidence < conf_threshold. }"
"{ top_k            | 1                                             | Usage: Keep top_k bounding boxes before NMS. }"
"{ save s           | true                                          | Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input. }"
"{ vis v            | true                                          | Usage: Specify to open a new window to show results. Invalid in case of camera input. }"
"{ backend bt       | 0                                             | Choose one of computation backends: "
"0: (default) OpenCV implementation + CPU, "
"1: CUDA + GPU (CUDA), "
"2: CUDA + GPU (CUDA FP16), "
"3: TIM-VX + NPU, "
"4: CANN + NPU}";


void drawLines(Mat image, Mat landmarks, Mat keeplandmarks, bool isDrawPoint = true, int thickness = 2)
{
    
    vector<pair<int, int>> segment = {
        make_pair(0, 1), make_pair(1, 2), make_pair(2, 3), make_pair(3, 7),
        make_pair(0, 4), make_pair(4, 5), make_pair(5, 6), make_pair(6, 8),
        make_pair(9, 10),
        make_pair(12, 14), make_pair(14, 16), make_pair(16, 22), make_pair(16, 18), make_pair(16, 20), make_pair(18, 20),
        make_pair(11, 13), make_pair(13, 15), make_pair(15, 21), make_pair(15, 19), make_pair(15, 17), make_pair(17, 19),
        make_pair(11, 12), make_pair(11, 23), make_pair(23, 24), make_pair(24, 12),
        make_pair(24, 26), make_pair(26, 28), make_pair(28, 30), make_pair(28, 32), make_pair(30, 32),
        make_pair(23, 25), make_pair(25, 27),make_pair(27, 31), make_pair(27, 29), make_pair(29, 31) };
    for (auto p : segment)
        if (keeplandmarks.at<uchar>(p.first) && keeplandmarks.at<uchar>(p.second))
            line(image, Point(landmarks.row(p.first)), Point(landmarks.row(p.second)), Scalar(255, 255, 255), thickness);
    if (isDrawPoint)
        for (int idxRow = 0; idxRow < landmarks.rows; idxRow++)
            if (keeplandmarks.at<uchar>(idxRow))
                circle(image, Point(landmarks.row(idxRow)), thickness, Scalar(0, 0, 255), -1);
}


pair<Mat, Mat> visualize(Mat image, vector<tuple<Mat, Mat, Mat, Mat, Mat, float>> poses, float fps=-1)
{
    Mat displayScreen = image.clone();
    Mat display3d(400, 400, CV_8UC3, Scalar::all(0));
    line(display3d, Point(200, 0), Point(200, 400), Scalar(255, 255, 255), 2);
    line(display3d, Point(0, 200), Point(400, 200), Scalar(255, 255, 255), 2);
    putText(display3d, "Main View", Point(0, 12), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    putText(display3d, "Top View", Point(200, 12), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    putText(display3d, "Left View", Point(0, 212), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    putText(display3d, "Right View", Point(200, 212), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    bool isDraw = false;  // ensure only one person is drawn

    for (auto pose : poses)
    {
        Mat bbox = get<0>(pose);
        if (!bbox.empty())
        {
            Mat landmarksScreen = get<1>(pose);
            Mat landmarksWord = get<2>(pose);
            Mat mask;
            get<3>(pose).convertTo(mask, CV_8U);
            Mat heatmap = get<4>(pose);
            float conf = get<5>(pose);
            Mat edges;
            Canny(mask, edges, 100, 200);
            Mat kernel(2, 2, CV_8UC1, Scalar::all(1)); // expansion edge to 2 pixels
            dilate(edges, edges, kernel);
            Mat edgesBGR;
            cvtColor(edges, edgesBGR, COLOR_GRAY2BGR);
            Mat idxSelec = edges == 255;
            edgesBGR.setTo(Scalar(0, 255, 0), idxSelec);

            add(edgesBGR, displayScreen, displayScreen);
            // draw box
            Mat box;
            bbox.convertTo(box, CV_32S);

            rectangle(displayScreen, Point(box.row(0)), Point(box.row(1)), Scalar(0, 255, 0), 2);
            putText(displayScreen, format("Conf = %4f", conf), Point(0, 35), FONT_HERSHEY_DUPLEX, 0.7,Scalar(0, 0, 255), 2);
            if (fps > 0)
                putText(displayScreen, format("FPS = %.2f", fps), Point(0, 55), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            // Draw line between each key points
            landmarksScreen = landmarksScreen.rowRange(0, landmarksScreen.rows - 6);
            landmarksWord = landmarksWord.rowRange(0, landmarksWord.rows - 6);

            Mat keepLandmarks = landmarksScreen.col(4) > 0.8; // only show visible keypoints which presence bigger than 0.8

            Mat landmarksXY;
            landmarksScreen.colRange(0, 2).convertTo(landmarksXY, CV_32S);
            drawLines(displayScreen, landmarksXY, keepLandmarks, false);

            // z value is relative to HIP, but we use constant to instead
            for (int idxRow = 0; idxRow < landmarksScreen.rows; idxRow++)
            {
                Mat landmark;// p in enumerate(landmarks_screen[:, 0 : 3].astype(np.int32))
                landmarksScreen.row(idxRow).convertTo(landmark, CV_32S);
                if (keepLandmarks.at<uchar>(idxRow))
                    circle(displayScreen, Point(landmark.at<int>(0), landmark.at<int>(1)), 2, Scalar(0, 0, 255), -1);
            }

            if (!isDraw)
            {
                isDraw = true;
                // Main view
                Mat landmarksXY = landmarksWord.colRange(0, 2).clone();
                Mat x = landmarksXY * 100 + 100;
                x.convertTo(landmarksXY, CV_32S);
                drawLines(display3d, landmarksXY, keepLandmarks, true, 2);

                // Top view
                Mat landmarksXZ;
                hconcat(landmarksWord.col(0), landmarksWord.col(2), landmarksXZ);
                landmarksXZ.col(1) = -landmarksXZ.col(1);
                x = landmarksXZ * 100;
                x.col(0) += 300;
                x.col(1) += 100;
                x.convertTo(landmarksXZ, CV_32S);
                drawLines(display3d, landmarksXZ, keepLandmarks, true, 2);

                // Left view
                Mat landmarksYZ;
                hconcat(landmarksWord.col(2), landmarksWord.col(1), landmarksYZ);
                landmarksYZ.col(0) = -landmarksYZ.col(0);
                x = landmarksYZ * 100;
                x.col(0) += 100;
                x.col(1) += 300;
                x.convertTo(landmarksYZ, CV_32S);
                drawLines(display3d, landmarksYZ, keepLandmarks, true, 2);

                // Right view
                Mat landmarksZY;
                hconcat(landmarksWord.col(2), landmarksWord.col(1), landmarksZY);
                x = landmarksZY * 100;
                x.col(0) += 300;
                x.col(1) += 300;
                x.convertTo(landmarksZY, CV_32S);
                drawLines(display3d, landmarksZY, keepLandmarks, true, 2);
            }
        }
    }
    return pair<Mat, Mat>(displayScreen, display3d);
}



int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
 
    parser.about("Person Detector from MediaPipe");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string model = parser.get<String>("model");
    float confThreshold = parser.get<float>("conf_threshold");
    float scoreThreshold = 0.5f;
    float nmsThreshold = 0.3f;
    int topK = 5000;
    bool vis = parser.get<bool>("vis");
    bool save = parser.get<bool>("save");
    int backendTargetid = parser.get<int>("backend");

    if (model.empty())
    {
        CV_Error(Error::StsError, "Model file " + model + " not found");
    }
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(samples::findFile(parser.get<String>("input")));
    else
        cap.open(0);
    Mat frame;
    // person detector
    MPPersonDet modelNet("../person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx", nmsThreshold, scoreThreshold, topK,
        backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);
    // pose estimator
    MPPose poseEstimator(model, confThreshold, backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);
    //! [Open a video file or an image file or a camera stream]
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot open video or file");

    static const std::string kWinName = "MPPose Demo";
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            if (parser.has("input"))
            {
                cout << "Frame is empty" << endl;
                break;
            }
            else
                continue;
        }
        TickMeter tm;
        tm.start();
        Mat person = modelNet.infer(frame);
        tm.stop();
        vector<tuple<Mat, Mat, Mat, Mat, Mat, float>> pose;
        for (int idxRow = 0; idxRow < person.rows; idxRow++)
        {
            tuple<Mat, Mat, Mat, Mat, Mat, float> re = poseEstimator.infer(frame, person.row(idxRow));
            if (!get<0>(re).empty())
                pose.push_back(re);
        }
        cout << "Inference time: " << tm.getTimeMilli() << " ms\n";
        pair<Mat, Mat> duoimg = visualize(frame, pose, tm.getFPS());
        if (vis)
        {
            imshow(kWinName, get<0>(duoimg));
            imshow("3d", get<1>(duoimg));
        }
    }
    return 0;
}


Mat getMediapipeAnchor()
{
    Mat anchor= (Mat_<float>(2254,2) << 0.017857142857142856, 0.017857142857142856,
        0.017857142857142856, 0.017857142857142856,
        0.05357142857142857, 0.017857142857142856,
        0.05357142857142857, 0.017857142857142856,
        0.08928571428571429, 0.017857142857142856,
        0.08928571428571429, 0.017857142857142856,
        0.125, 0.017857142857142856,
        0.125, 0.017857142857142856,
        0.16071428571428573, 0.017857142857142856,
        0.16071428571428573, 0.017857142857142856,
        0.19642857142857142, 0.017857142857142856,
        0.19642857142857142, 0.017857142857142856,
        0.23214285714285715, 0.017857142857142856,
        0.23214285714285715, 0.017857142857142856,
        0.26785714285714285, 0.017857142857142856,
        0.26785714285714285, 0.017857142857142856,
        0.30357142857142855, 0.017857142857142856,
        0.30357142857142855, 0.017857142857142856,
        0.3392857142857143, 0.017857142857142856,
        0.3392857142857143, 0.017857142857142856,
        0.375, 0.017857142857142856,
        0.375, 0.017857142857142856,
        0.4107142857142857, 0.017857142857142856,
        0.4107142857142857, 0.017857142857142856,
        0.44642857142857145, 0.017857142857142856,
        0.44642857142857145, 0.017857142857142856,
        0.48214285714285715, 0.017857142857142856,
        0.48214285714285715, 0.017857142857142856,
        0.5178571428571429, 0.017857142857142856,
        0.5178571428571429, 0.017857142857142856,
        0.5535714285714286, 0.017857142857142856,
        0.5535714285714286, 0.017857142857142856,
        0.5892857142857143, 0.017857142857142856,
        0.5892857142857143, 0.017857142857142856,
        0.625, 0.017857142857142856,
        0.625, 0.017857142857142856,
        0.6607142857142857, 0.017857142857142856,
        0.6607142857142857, 0.017857142857142856,
        0.6964285714285714, 0.017857142857142856,
        0.6964285714285714, 0.017857142857142856,
        0.7321428571428571, 0.017857142857142856,
        0.7321428571428571, 0.017857142857142856,
        0.7678571428571429, 0.017857142857142856,
        0.7678571428571429, 0.017857142857142856,
        0.8035714285714286, 0.017857142857142856,
        0.8035714285714286, 0.017857142857142856,
        0.8392857142857143, 0.017857142857142856,
        0.8392857142857143, 0.017857142857142856,
        0.875, 0.017857142857142856,
        0.875, 0.017857142857142856,
        0.9107142857142857, 0.017857142857142856,
        0.9107142857142857, 0.017857142857142856,
        0.9464285714285714, 0.017857142857142856,
        0.9464285714285714, 0.017857142857142856,
        0.9821428571428571, 0.017857142857142856,
        0.9821428571428571, 0.017857142857142856,
        0.017857142857142856, 0.05357142857142857,
        0.017857142857142856, 0.05357142857142857,
        0.05357142857142857, 0.05357142857142857,
        0.05357142857142857, 0.05357142857142857,
        0.08928571428571429, 0.05357142857142857,
        0.08928571428571429, 0.05357142857142857,
        0.125, 0.05357142857142857,
        0.125, 0.05357142857142857,
        0.16071428571428573, 0.05357142857142857,
        0.16071428571428573, 0.05357142857142857,
        0.19642857142857142, 0.05357142857142857,
        0.19642857142857142, 0.05357142857142857,
        0.23214285714285715, 0.05357142857142857,
        0.23214285714285715, 0.05357142857142857,
        0.26785714285714285, 0.05357142857142857,
        0.26785714285714285, 0.05357142857142857,
        0.30357142857142855, 0.05357142857142857,
        0.30357142857142855, 0.05357142857142857,
        0.3392857142857143, 0.05357142857142857,
        0.3392857142857143, 0.05357142857142857,
        0.375, 0.05357142857142857,
        0.375, 0.05357142857142857,
        0.4107142857142857, 0.05357142857142857,
        0.4107142857142857, 0.05357142857142857,
        0.44642857142857145, 0.05357142857142857,
        0.44642857142857145, 0.05357142857142857,
        0.48214285714285715, 0.05357142857142857,
        0.48214285714285715, 0.05357142857142857,
        0.5178571428571429, 0.05357142857142857,
        0.5178571428571429, 0.05357142857142857,
        0.5535714285714286, 0.05357142857142857,
        0.5535714285714286, 0.05357142857142857,
        0.5892857142857143, 0.05357142857142857,
        0.5892857142857143, 0.05357142857142857,
        0.625, 0.05357142857142857,
        0.625, 0.05357142857142857,
        0.6607142857142857, 0.05357142857142857,
        0.6607142857142857, 0.05357142857142857,
        0.6964285714285714, 0.05357142857142857,
        0.6964285714285714, 0.05357142857142857,
        0.7321428571428571, 0.05357142857142857,
        0.7321428571428571, 0.05357142857142857,
        0.7678571428571429, 0.05357142857142857,
        0.7678571428571429, 0.05357142857142857,
        0.8035714285714286, 0.05357142857142857,
        0.8035714285714286, 0.05357142857142857,
        0.8392857142857143, 0.05357142857142857,
        0.8392857142857143, 0.05357142857142857,
        0.875, 0.05357142857142857,
        0.875, 0.05357142857142857,
        0.9107142857142857, 0.05357142857142857,
        0.9107142857142857, 0.05357142857142857,
        0.9464285714285714, 0.05357142857142857,
        0.9464285714285714, 0.05357142857142857,
        0.9821428571428571, 0.05357142857142857,
        0.9821428571428571, 0.05357142857142857,
        0.017857142857142856, 0.08928571428571429,
        0.017857142857142856, 0.08928571428571429,
        0.05357142857142857, 0.08928571428571429,
        0.05357142857142857, 0.08928571428571429,
        0.08928571428571429, 0.08928571428571429,
        0.08928571428571429, 0.08928571428571429,
        0.125, 0.08928571428571429,
        0.125, 0.08928571428571429,
        0.16071428571428573, 0.08928571428571429,
        0.16071428571428573, 0.08928571428571429,
        0.19642857142857142, 0.08928571428571429,
        0.19642857142857142, 0.08928571428571429,
        0.23214285714285715, 0.08928571428571429,
        0.23214285714285715, 0.08928571428571429,
        0.26785714285714285, 0.08928571428571429,
        0.26785714285714285, 0.08928571428571429,
        0.30357142857142855, 0.08928571428571429,
        0.30357142857142855, 0.08928571428571429,
        0.3392857142857143, 0.08928571428571429,
        0.3392857142857143, 0.08928571428571429,
        0.375, 0.08928571428571429,
        0.375, 0.08928571428571429,
        0.4107142857142857, 0.08928571428571429,
        0.4107142857142857, 0.08928571428571429,
        0.44642857142857145, 0.08928571428571429,
        0.44642857142857145, 0.08928571428571429,
        0.48214285714285715, 0.08928571428571429,
        0.48214285714285715, 0.08928571428571429,
        0.5178571428571429, 0.08928571428571429,
        0.5178571428571429, 0.08928571428571429,
        0.5535714285714286, 0.08928571428571429,
        0.5535714285714286, 0.08928571428571429,
        0.5892857142857143, 0.08928571428571429,
        0.5892857142857143, 0.08928571428571429,
        0.625, 0.08928571428571429,
        0.625, 0.08928571428571429,
        0.6607142857142857, 0.08928571428571429,
        0.6607142857142857, 0.08928571428571429,
        0.6964285714285714, 0.08928571428571429,
        0.6964285714285714, 0.08928571428571429,
        0.7321428571428571, 0.08928571428571429,
        0.7321428571428571, 0.08928571428571429,
        0.7678571428571429, 0.08928571428571429,
        0.7678571428571429, 0.08928571428571429,
        0.8035714285714286, 0.08928571428571429,
        0.8035714285714286, 0.08928571428571429,
        0.8392857142857143, 0.08928571428571429,
        0.8392857142857143, 0.08928571428571429,
        0.875, 0.08928571428571429,
        0.875, 0.08928571428571429,
        0.9107142857142857, 0.08928571428571429,
        0.9107142857142857, 0.08928571428571429,
        0.9464285714285714, 0.08928571428571429,
        0.9464285714285714, 0.08928571428571429,
        0.9821428571428571, 0.08928571428571429,
        0.9821428571428571, 0.08928571428571429,
        0.017857142857142856, 0.125,
        0.017857142857142856, 0.125,
        0.05357142857142857, 0.125,
        0.05357142857142857, 0.125,
        0.08928571428571429, 0.125,
        0.08928571428571429, 0.125,
        0.125, 0.125,
        0.125, 0.125,
        0.16071428571428573, 0.125,
        0.16071428571428573, 0.125,
        0.19642857142857142, 0.125,
        0.19642857142857142, 0.125,
        0.23214285714285715, 0.125,
        0.23214285714285715, 0.125,
        0.26785714285714285, 0.125,
        0.26785714285714285, 0.125,
        0.30357142857142855, 0.125,
        0.30357142857142855, 0.125,
        0.3392857142857143, 0.125,
        0.3392857142857143, 0.125,
        0.375, 0.125,
        0.375, 0.125,
        0.4107142857142857, 0.125,
        0.4107142857142857, 0.125,
        0.44642857142857145, 0.125,
        0.44642857142857145, 0.125,
        0.48214285714285715, 0.125,
        0.48214285714285715, 0.125,
        0.5178571428571429, 0.125,
        0.5178571428571429, 0.125,
        0.5535714285714286, 0.125,
        0.5535714285714286, 0.125,
        0.5892857142857143, 0.125,
        0.5892857142857143, 0.125,
        0.625, 0.125,
        0.625, 0.125,
        0.6607142857142857, 0.125,
        0.6607142857142857, 0.125,
        0.6964285714285714, 0.125,
        0.6964285714285714, 0.125,
        0.7321428571428571, 0.125,
        0.7321428571428571, 0.125,
        0.7678571428571429, 0.125,
        0.7678571428571429, 0.125,
        0.8035714285714286, 0.125,
        0.8035714285714286, 0.125,
        0.8392857142857143, 0.125,
        0.8392857142857143, 0.125,
        0.875, 0.125,
        0.875, 0.125,
        0.9107142857142857, 0.125,
        0.9107142857142857, 0.125,
        0.9464285714285714, 0.125,
        0.9464285714285714, 0.125,
        0.9821428571428571, 0.125,
        0.9821428571428571, 0.125,
        0.017857142857142856, 0.16071428571428573,
        0.017857142857142856, 0.16071428571428573,
        0.05357142857142857, 0.16071428571428573,
        0.05357142857142857, 0.16071428571428573,
        0.08928571428571429, 0.16071428571428573,
        0.08928571428571429, 0.16071428571428573,
        0.125, 0.16071428571428573,
        0.125, 0.16071428571428573,
        0.16071428571428573, 0.16071428571428573,
        0.16071428571428573, 0.16071428571428573,
        0.19642857142857142, 0.16071428571428573,
        0.19642857142857142, 0.16071428571428573,
        0.23214285714285715, 0.16071428571428573,
        0.23214285714285715, 0.16071428571428573,
        0.26785714285714285, 0.16071428571428573,
        0.26785714285714285, 0.16071428571428573,
        0.30357142857142855, 0.16071428571428573,
        0.30357142857142855, 0.16071428571428573,
        0.3392857142857143, 0.16071428571428573,
        0.3392857142857143, 0.16071428571428573,
        0.375, 0.16071428571428573,
        0.375, 0.16071428571428573,
        0.4107142857142857, 0.16071428571428573,
        0.4107142857142857, 0.16071428571428573,
        0.44642857142857145, 0.16071428571428573,
        0.44642857142857145, 0.16071428571428573,
        0.48214285714285715, 0.16071428571428573,
        0.48214285714285715, 0.16071428571428573,
        0.5178571428571429, 0.16071428571428573,
        0.5178571428571429, 0.16071428571428573,
        0.5535714285714286, 0.16071428571428573,
        0.5535714285714286, 0.16071428571428573,
        0.5892857142857143, 0.16071428571428573,
        0.5892857142857143, 0.16071428571428573,
        0.625, 0.16071428571428573,
        0.625, 0.16071428571428573,
        0.6607142857142857, 0.16071428571428573,
        0.6607142857142857, 0.16071428571428573,
        0.6964285714285714, 0.16071428571428573,
        0.6964285714285714, 0.16071428571428573,
        0.7321428571428571, 0.16071428571428573,
        0.7321428571428571, 0.16071428571428573,
        0.7678571428571429, 0.16071428571428573,
        0.7678571428571429, 0.16071428571428573,
        0.8035714285714286, 0.16071428571428573,
        0.8035714285714286, 0.16071428571428573,
        0.8392857142857143, 0.16071428571428573,
        0.8392857142857143, 0.16071428571428573,
        0.875, 0.16071428571428573,
        0.875, 0.16071428571428573,
        0.9107142857142857, 0.16071428571428573,
        0.9107142857142857, 0.16071428571428573,
        0.9464285714285714, 0.16071428571428573,
        0.9464285714285714, 0.16071428571428573,
        0.9821428571428571, 0.16071428571428573,
        0.9821428571428571, 0.16071428571428573,
        0.017857142857142856, 0.19642857142857142,
        0.017857142857142856, 0.19642857142857142,
        0.05357142857142857, 0.19642857142857142,
        0.05357142857142857, 0.19642857142857142,
        0.08928571428571429, 0.19642857142857142,
        0.08928571428571429, 0.19642857142857142,
        0.125, 0.19642857142857142,
        0.125, 0.19642857142857142,
        0.16071428571428573, 0.19642857142857142,
        0.16071428571428573, 0.19642857142857142,
        0.19642857142857142, 0.19642857142857142,
        0.19642857142857142, 0.19642857142857142,
        0.23214285714285715, 0.19642857142857142,
        0.23214285714285715, 0.19642857142857142,
        0.26785714285714285, 0.19642857142857142,
        0.26785714285714285, 0.19642857142857142,
        0.30357142857142855, 0.19642857142857142,
        0.30357142857142855, 0.19642857142857142,
        0.3392857142857143, 0.19642857142857142,
        0.3392857142857143, 0.19642857142857142,
        0.375, 0.19642857142857142,
        0.375, 0.19642857142857142,
        0.4107142857142857, 0.19642857142857142,
        0.4107142857142857, 0.19642857142857142,
        0.44642857142857145, 0.19642857142857142,
        0.44642857142857145, 0.19642857142857142,
        0.48214285714285715, 0.19642857142857142,
        0.48214285714285715, 0.19642857142857142,
        0.5178571428571429, 0.19642857142857142,
        0.5178571428571429, 0.19642857142857142,
        0.5535714285714286, 0.19642857142857142,
        0.5535714285714286, 0.19642857142857142,
        0.5892857142857143, 0.19642857142857142,
        0.5892857142857143, 0.19642857142857142,
        0.625, 0.19642857142857142,
        0.625, 0.19642857142857142,
        0.6607142857142857, 0.19642857142857142,
        0.6607142857142857, 0.19642857142857142,
        0.6964285714285714, 0.19642857142857142,
        0.6964285714285714, 0.19642857142857142,
        0.7321428571428571, 0.19642857142857142,
        0.7321428571428571, 0.19642857142857142,
        0.7678571428571429, 0.19642857142857142,
        0.7678571428571429, 0.19642857142857142,
        0.8035714285714286, 0.19642857142857142,
        0.8035714285714286, 0.19642857142857142,
        0.8392857142857143, 0.19642857142857142,
        0.8392857142857143, 0.19642857142857142,
        0.875, 0.19642857142857142,
        0.875, 0.19642857142857142,
        0.9107142857142857, 0.19642857142857142,
        0.9107142857142857, 0.19642857142857142,
        0.9464285714285714, 0.19642857142857142,
        0.9464285714285714, 0.19642857142857142,
        0.9821428571428571, 0.19642857142857142,
        0.9821428571428571, 0.19642857142857142,
        0.017857142857142856, 0.23214285714285715,
        0.017857142857142856, 0.23214285714285715,
        0.05357142857142857, 0.23214285714285715,
        0.05357142857142857, 0.23214285714285715,
        0.08928571428571429, 0.23214285714285715,
        0.08928571428571429, 0.23214285714285715,
        0.125, 0.23214285714285715,
        0.125, 0.23214285714285715,
        0.16071428571428573, 0.23214285714285715,
        0.16071428571428573, 0.23214285714285715,
        0.19642857142857142, 0.23214285714285715,
        0.19642857142857142, 0.23214285714285715,
        0.23214285714285715, 0.23214285714285715,
        0.23214285714285715, 0.23214285714285715,
        0.26785714285714285, 0.23214285714285715,
        0.26785714285714285, 0.23214285714285715,
        0.30357142857142855, 0.23214285714285715,
        0.30357142857142855, 0.23214285714285715,
        0.3392857142857143, 0.23214285714285715,
        0.3392857142857143, 0.23214285714285715,
        0.375, 0.23214285714285715,
        0.375, 0.23214285714285715,
        0.4107142857142857, 0.23214285714285715,
        0.4107142857142857, 0.23214285714285715,
        0.44642857142857145, 0.23214285714285715,
        0.44642857142857145, 0.23214285714285715,
        0.48214285714285715, 0.23214285714285715,
        0.48214285714285715, 0.23214285714285715,
        0.5178571428571429, 0.23214285714285715,
        0.5178571428571429, 0.23214285714285715,
        0.5535714285714286, 0.23214285714285715,
        0.5535714285714286, 0.23214285714285715,
        0.5892857142857143, 0.23214285714285715,
        0.5892857142857143, 0.23214285714285715,
        0.625, 0.23214285714285715,
        0.625, 0.23214285714285715,
        0.6607142857142857, 0.23214285714285715,
        0.6607142857142857, 0.23214285714285715,
        0.6964285714285714, 0.23214285714285715,
        0.6964285714285714, 0.23214285714285715,
        0.7321428571428571, 0.23214285714285715,
        0.7321428571428571, 0.23214285714285715,
        0.7678571428571429, 0.23214285714285715,
        0.7678571428571429, 0.23214285714285715,
        0.8035714285714286, 0.23214285714285715,
        0.8035714285714286, 0.23214285714285715,
        0.8392857142857143, 0.23214285714285715,
        0.8392857142857143, 0.23214285714285715,
        0.875, 0.23214285714285715,
        0.875, 0.23214285714285715,
        0.9107142857142857, 0.23214285714285715,
        0.9107142857142857, 0.23214285714285715,
        0.9464285714285714, 0.23214285714285715,
        0.9464285714285714, 0.23214285714285715,
        0.9821428571428571, 0.23214285714285715,
        0.9821428571428571, 0.23214285714285715,
        0.017857142857142856, 0.26785714285714285,
        0.017857142857142856, 0.26785714285714285,
        0.05357142857142857, 0.26785714285714285,
        0.05357142857142857, 0.26785714285714285,
        0.08928571428571429, 0.26785714285714285,
        0.08928571428571429, 0.26785714285714285,
        0.125, 0.26785714285714285,
        0.125, 0.26785714285714285,
        0.16071428571428573, 0.26785714285714285,
        0.16071428571428573, 0.26785714285714285,
        0.19642857142857142, 0.26785714285714285,
        0.19642857142857142, 0.26785714285714285,
        0.23214285714285715, 0.26785714285714285,
        0.23214285714285715, 0.26785714285714285,
        0.26785714285714285, 0.26785714285714285,
        0.26785714285714285, 0.26785714285714285,
        0.30357142857142855, 0.26785714285714285,
        0.30357142857142855, 0.26785714285714285,
        0.3392857142857143, 0.26785714285714285,
        0.3392857142857143, 0.26785714285714285,
        0.375, 0.26785714285714285,
        0.375, 0.26785714285714285,
        0.4107142857142857, 0.26785714285714285,
        0.4107142857142857, 0.26785714285714285,
        0.44642857142857145, 0.26785714285714285,
        0.44642857142857145, 0.26785714285714285,
        0.48214285714285715, 0.26785714285714285,
        0.48214285714285715, 0.26785714285714285,
        0.5178571428571429, 0.26785714285714285,
        0.5178571428571429, 0.26785714285714285,
        0.5535714285714286, 0.26785714285714285,
        0.5535714285714286, 0.26785714285714285,
        0.5892857142857143, 0.26785714285714285,
        0.5892857142857143, 0.26785714285714285,
        0.625, 0.26785714285714285,
        0.625, 0.26785714285714285,
        0.6607142857142857, 0.26785714285714285,
        0.6607142857142857, 0.26785714285714285,
        0.6964285714285714, 0.26785714285714285,
        0.6964285714285714, 0.26785714285714285,
        0.7321428571428571, 0.26785714285714285,
        0.7321428571428571, 0.26785714285714285,
        0.7678571428571429, 0.26785714285714285,
        0.7678571428571429, 0.26785714285714285,
        0.8035714285714286, 0.26785714285714285,
        0.8035714285714286, 0.26785714285714285,
        0.8392857142857143, 0.26785714285714285,
        0.8392857142857143, 0.26785714285714285,
        0.875, 0.26785714285714285,
        0.875, 0.26785714285714285,
        0.9107142857142857, 0.26785714285714285,
        0.9107142857142857, 0.26785714285714285,
        0.9464285714285714, 0.26785714285714285,
        0.9464285714285714, 0.26785714285714285,
        0.9821428571428571, 0.26785714285714285,
        0.9821428571428571, 0.26785714285714285,
        0.017857142857142856, 0.30357142857142855,
        0.017857142857142856, 0.30357142857142855,
        0.05357142857142857, 0.30357142857142855,
        0.05357142857142857, 0.30357142857142855,
        0.08928571428571429, 0.30357142857142855,
        0.08928571428571429, 0.30357142857142855,
        0.125, 0.30357142857142855,
        0.125, 0.30357142857142855,
        0.16071428571428573, 0.30357142857142855,
        0.16071428571428573, 0.30357142857142855,
        0.19642857142857142, 0.30357142857142855,
        0.19642857142857142, 0.30357142857142855,
        0.23214285714285715, 0.30357142857142855,
        0.23214285714285715, 0.30357142857142855,
        0.26785714285714285, 0.30357142857142855,
        0.26785714285714285, 0.30357142857142855,
        0.30357142857142855, 0.30357142857142855,
        0.30357142857142855, 0.30357142857142855,
        0.3392857142857143, 0.30357142857142855,
        0.3392857142857143, 0.30357142857142855,
        0.375, 0.30357142857142855,
        0.375, 0.30357142857142855,
        0.4107142857142857, 0.30357142857142855,
        0.4107142857142857, 0.30357142857142855,
        0.44642857142857145, 0.30357142857142855,
        0.44642857142857145, 0.30357142857142855,
        0.48214285714285715, 0.30357142857142855,
        0.48214285714285715, 0.30357142857142855,
        0.5178571428571429, 0.30357142857142855,
        0.5178571428571429, 0.30357142857142855,
        0.5535714285714286, 0.30357142857142855,
        0.5535714285714286, 0.30357142857142855,
        0.5892857142857143, 0.30357142857142855,
        0.5892857142857143, 0.30357142857142855,
        0.625, 0.30357142857142855,
        0.625, 0.30357142857142855,
        0.6607142857142857, 0.30357142857142855,
        0.6607142857142857, 0.30357142857142855,
        0.6964285714285714, 0.30357142857142855,
        0.6964285714285714, 0.30357142857142855,
        0.7321428571428571, 0.30357142857142855,
        0.7321428571428571, 0.30357142857142855,
        0.7678571428571429, 0.30357142857142855,
        0.7678571428571429, 0.30357142857142855,
        0.8035714285714286, 0.30357142857142855,
        0.8035714285714286, 0.30357142857142855,
        0.8392857142857143, 0.30357142857142855,
        0.8392857142857143, 0.30357142857142855,
        0.875, 0.30357142857142855,
        0.875, 0.30357142857142855,
        0.9107142857142857, 0.30357142857142855,
        0.9107142857142857, 0.30357142857142855,
        0.9464285714285714, 0.30357142857142855,
        0.9464285714285714, 0.30357142857142855,
        0.9821428571428571, 0.30357142857142855,
        0.9821428571428571, 0.30357142857142855,
        0.017857142857142856, 0.3392857142857143,
        0.017857142857142856, 0.3392857142857143,
        0.05357142857142857, 0.3392857142857143,
        0.05357142857142857, 0.3392857142857143,
        0.08928571428571429, 0.3392857142857143,
        0.08928571428571429, 0.3392857142857143,
        0.125, 0.3392857142857143,
        0.125, 0.3392857142857143,
        0.16071428571428573, 0.3392857142857143,
        0.16071428571428573, 0.3392857142857143,
        0.19642857142857142, 0.3392857142857143,
        0.19642857142857142, 0.3392857142857143,
        0.23214285714285715, 0.3392857142857143,
        0.23214285714285715, 0.3392857142857143,
        0.26785714285714285, 0.3392857142857143,
        0.26785714285714285, 0.3392857142857143,
        0.30357142857142855, 0.3392857142857143,
        0.30357142857142855, 0.3392857142857143,
        0.3392857142857143, 0.3392857142857143,
        0.3392857142857143, 0.3392857142857143,
        0.375, 0.3392857142857143,
        0.375, 0.3392857142857143,
        0.4107142857142857, 0.3392857142857143,
        0.4107142857142857, 0.3392857142857143,
        0.44642857142857145, 0.3392857142857143,
        0.44642857142857145, 0.3392857142857143,
        0.48214285714285715, 0.3392857142857143,
        0.48214285714285715, 0.3392857142857143,
        0.5178571428571429, 0.3392857142857143,
        0.5178571428571429, 0.3392857142857143,
        0.5535714285714286, 0.3392857142857143,
        0.5535714285714286, 0.3392857142857143,
        0.5892857142857143, 0.3392857142857143,
        0.5892857142857143, 0.3392857142857143,
        0.625, 0.3392857142857143,
        0.625, 0.3392857142857143,
        0.6607142857142857, 0.3392857142857143,
        0.6607142857142857, 0.3392857142857143,
        0.6964285714285714, 0.3392857142857143,
        0.6964285714285714, 0.3392857142857143,
        0.7321428571428571, 0.3392857142857143,
        0.7321428571428571, 0.3392857142857143,
        0.7678571428571429, 0.3392857142857143,
        0.7678571428571429, 0.3392857142857143,
        0.8035714285714286, 0.3392857142857143,
        0.8035714285714286, 0.3392857142857143,
        0.8392857142857143, 0.3392857142857143,
        0.8392857142857143, 0.3392857142857143,
        0.875, 0.3392857142857143,
        0.875, 0.3392857142857143,
        0.9107142857142857, 0.3392857142857143,
        0.9107142857142857, 0.3392857142857143,
        0.9464285714285714, 0.3392857142857143,
        0.9464285714285714, 0.3392857142857143,
        0.9821428571428571, 0.3392857142857143,
        0.9821428571428571, 0.3392857142857143,
        0.017857142857142856, 0.375,
        0.017857142857142856, 0.375,
        0.05357142857142857, 0.375,
        0.05357142857142857, 0.375,
        0.08928571428571429, 0.375,
        0.08928571428571429, 0.375,
        0.125, 0.375,
        0.125, 0.375,
        0.16071428571428573, 0.375,
        0.16071428571428573, 0.375,
        0.19642857142857142, 0.375,
        0.19642857142857142, 0.375,
        0.23214285714285715, 0.375,
        0.23214285714285715, 0.375,
        0.26785714285714285, 0.375,
        0.26785714285714285, 0.375,
        0.30357142857142855, 0.375,
        0.30357142857142855, 0.375,
        0.3392857142857143, 0.375,
        0.3392857142857143, 0.375,
        0.375, 0.375,
        0.375, 0.375,
        0.4107142857142857, 0.375,
        0.4107142857142857, 0.375,
        0.44642857142857145, 0.375,
        0.44642857142857145, 0.375,
        0.48214285714285715, 0.375,
        0.48214285714285715, 0.375,
        0.5178571428571429, 0.375,
        0.5178571428571429, 0.375,
        0.5535714285714286, 0.375,
        0.5535714285714286, 0.375,
        0.5892857142857143, 0.375,
        0.5892857142857143, 0.375,
        0.625, 0.375,
        0.625, 0.375,
        0.6607142857142857, 0.375,
        0.6607142857142857, 0.375,
        0.6964285714285714, 0.375,
        0.6964285714285714, 0.375,
        0.7321428571428571, 0.375,
        0.7321428571428571, 0.375,
        0.7678571428571429, 0.375,
        0.7678571428571429, 0.375,
        0.8035714285714286, 0.375,
        0.8035714285714286, 0.375,
        0.8392857142857143, 0.375,
        0.8392857142857143, 0.375,
        0.875, 0.375,
        0.875, 0.375,
        0.9107142857142857, 0.375,
        0.9107142857142857, 0.375,
        0.9464285714285714, 0.375,
        0.9464285714285714, 0.375,
        0.9821428571428571, 0.375,
        0.9821428571428571, 0.375,
        0.017857142857142856, 0.4107142857142857,
        0.017857142857142856, 0.4107142857142857,
        0.05357142857142857, 0.4107142857142857,
        0.05357142857142857, 0.4107142857142857,
        0.08928571428571429, 0.4107142857142857,
        0.08928571428571429, 0.4107142857142857,
        0.125, 0.4107142857142857,
        0.125, 0.4107142857142857,
        0.16071428571428573, 0.4107142857142857,
        0.16071428571428573, 0.4107142857142857,
        0.19642857142857142, 0.4107142857142857,
        0.19642857142857142, 0.4107142857142857,
        0.23214285714285715, 0.4107142857142857,
        0.23214285714285715, 0.4107142857142857,
        0.26785714285714285, 0.4107142857142857,
        0.26785714285714285, 0.4107142857142857,
        0.30357142857142855, 0.4107142857142857,
        0.30357142857142855, 0.4107142857142857,
        0.3392857142857143, 0.4107142857142857,
        0.3392857142857143, 0.4107142857142857,
        0.375, 0.4107142857142857,
        0.375, 0.4107142857142857,
        0.4107142857142857, 0.4107142857142857,
        0.4107142857142857, 0.4107142857142857,
        0.44642857142857145, 0.4107142857142857,
        0.44642857142857145, 0.4107142857142857,
        0.48214285714285715, 0.4107142857142857,
        0.48214285714285715, 0.4107142857142857,
        0.5178571428571429, 0.4107142857142857,
        0.5178571428571429, 0.4107142857142857,
        0.5535714285714286, 0.4107142857142857,
        0.5535714285714286, 0.4107142857142857,
        0.5892857142857143, 0.4107142857142857,
        0.5892857142857143, 0.4107142857142857,
        0.625, 0.4107142857142857,
        0.625, 0.4107142857142857,
        0.6607142857142857, 0.4107142857142857,
        0.6607142857142857, 0.4107142857142857,
        0.6964285714285714, 0.4107142857142857,
        0.6964285714285714, 0.4107142857142857,
        0.7321428571428571, 0.4107142857142857,
        0.7321428571428571, 0.4107142857142857,
        0.7678571428571429, 0.4107142857142857,
        0.7678571428571429, 0.4107142857142857,
        0.8035714285714286, 0.4107142857142857,
        0.8035714285714286, 0.4107142857142857,
        0.8392857142857143, 0.4107142857142857,
        0.8392857142857143, 0.4107142857142857,
        0.875, 0.4107142857142857,
        0.875, 0.4107142857142857,
        0.9107142857142857, 0.4107142857142857,
        0.9107142857142857, 0.4107142857142857,
        0.9464285714285714, 0.4107142857142857,
        0.9464285714285714, 0.4107142857142857,
        0.9821428571428571, 0.4107142857142857,
        0.9821428571428571, 0.4107142857142857,
        0.017857142857142856, 0.44642857142857145,
        0.017857142857142856, 0.44642857142857145,
        0.05357142857142857, 0.44642857142857145,
        0.05357142857142857, 0.44642857142857145,
        0.08928571428571429, 0.44642857142857145,
        0.08928571428571429, 0.44642857142857145,
        0.125, 0.44642857142857145,
        0.125, 0.44642857142857145,
        0.16071428571428573, 0.44642857142857145,
        0.16071428571428573, 0.44642857142857145,
        0.19642857142857142, 0.44642857142857145,
        0.19642857142857142, 0.44642857142857145,
        0.23214285714285715, 0.44642857142857145,
        0.23214285714285715, 0.44642857142857145,
        0.26785714285714285, 0.44642857142857145,
        0.26785714285714285, 0.44642857142857145,
        0.30357142857142855, 0.44642857142857145,
        0.30357142857142855, 0.44642857142857145,
        0.3392857142857143, 0.44642857142857145,
        0.3392857142857143, 0.44642857142857145,
        0.375, 0.44642857142857145,
        0.375, 0.44642857142857145,
        0.4107142857142857, 0.44642857142857145,
        0.4107142857142857, 0.44642857142857145,
        0.44642857142857145, 0.44642857142857145,
        0.44642857142857145, 0.44642857142857145,
        0.48214285714285715, 0.44642857142857145,
        0.48214285714285715, 0.44642857142857145,
        0.5178571428571429, 0.44642857142857145,
        0.5178571428571429, 0.44642857142857145,
        0.5535714285714286, 0.44642857142857145,
        0.5535714285714286, 0.44642857142857145,
        0.5892857142857143, 0.44642857142857145,
        0.5892857142857143, 0.44642857142857145,
        0.625, 0.44642857142857145,
        0.625, 0.44642857142857145,
        0.6607142857142857, 0.44642857142857145,
        0.6607142857142857, 0.44642857142857145,
        0.6964285714285714, 0.44642857142857145,
        0.6964285714285714, 0.44642857142857145,
        0.7321428571428571, 0.44642857142857145,
        0.7321428571428571, 0.44642857142857145,
        0.7678571428571429, 0.44642857142857145,
        0.7678571428571429, 0.44642857142857145,
        0.8035714285714286, 0.44642857142857145,
        0.8035714285714286, 0.44642857142857145,
        0.8392857142857143, 0.44642857142857145,
        0.8392857142857143, 0.44642857142857145,
        0.875, 0.44642857142857145,
        0.875, 0.44642857142857145,
        0.9107142857142857, 0.44642857142857145,
        0.9107142857142857, 0.44642857142857145,
        0.9464285714285714, 0.44642857142857145,
        0.9464285714285714, 0.44642857142857145,
        0.9821428571428571, 0.44642857142857145,
        0.9821428571428571, 0.44642857142857145,
        0.017857142857142856, 0.48214285714285715,
        0.017857142857142856, 0.48214285714285715,
        0.05357142857142857, 0.48214285714285715,
        0.05357142857142857, 0.48214285714285715,
        0.08928571428571429, 0.48214285714285715,
        0.08928571428571429, 0.48214285714285715,
        0.125, 0.48214285714285715,
        0.125, 0.48214285714285715,
        0.16071428571428573, 0.48214285714285715,
        0.16071428571428573, 0.48214285714285715,
        0.19642857142857142, 0.48214285714285715,
        0.19642857142857142, 0.48214285714285715,
        0.23214285714285715, 0.48214285714285715,
        0.23214285714285715, 0.48214285714285715,
        0.26785714285714285, 0.48214285714285715,
        0.26785714285714285, 0.48214285714285715,
        0.30357142857142855, 0.48214285714285715,
        0.30357142857142855, 0.48214285714285715,
        0.3392857142857143, 0.48214285714285715,
        0.3392857142857143, 0.48214285714285715,
        0.375, 0.48214285714285715,
        0.375, 0.48214285714285715,
        0.4107142857142857, 0.48214285714285715,
        0.4107142857142857, 0.48214285714285715,
        0.44642857142857145, 0.48214285714285715,
        0.44642857142857145, 0.48214285714285715,
        0.48214285714285715, 0.48214285714285715,
        0.48214285714285715, 0.48214285714285715,
        0.5178571428571429, 0.48214285714285715,
        0.5178571428571429, 0.48214285714285715,
        0.5535714285714286, 0.48214285714285715,
        0.5535714285714286, 0.48214285714285715,
        0.5892857142857143, 0.48214285714285715,
        0.5892857142857143, 0.48214285714285715,
        0.625, 0.48214285714285715,
        0.625, 0.48214285714285715,
        0.6607142857142857, 0.48214285714285715,
        0.6607142857142857, 0.48214285714285715,
        0.6964285714285714, 0.48214285714285715,
        0.6964285714285714, 0.48214285714285715,
        0.7321428571428571, 0.48214285714285715,
        0.7321428571428571, 0.48214285714285715,
        0.7678571428571429, 0.48214285714285715,
        0.7678571428571429, 0.48214285714285715,
        0.8035714285714286, 0.48214285714285715,
        0.8035714285714286, 0.48214285714285715,
        0.8392857142857143, 0.48214285714285715,
        0.8392857142857143, 0.48214285714285715,
        0.875, 0.48214285714285715,
        0.875, 0.48214285714285715,
        0.9107142857142857, 0.48214285714285715,
        0.9107142857142857, 0.48214285714285715,
        0.9464285714285714, 0.48214285714285715,
        0.9464285714285714, 0.48214285714285715,
        0.9821428571428571, 0.48214285714285715,
        0.9821428571428571, 0.48214285714285715,
        0.017857142857142856, 0.5178571428571429,
        0.017857142857142856, 0.5178571428571429,
        0.05357142857142857, 0.5178571428571429,
        0.05357142857142857, 0.5178571428571429,
        0.08928571428571429, 0.5178571428571429,
        0.08928571428571429, 0.5178571428571429,
        0.125, 0.5178571428571429,
        0.125, 0.5178571428571429,
        0.16071428571428573, 0.5178571428571429,
        0.16071428571428573, 0.5178571428571429,
        0.19642857142857142, 0.5178571428571429,
        0.19642857142857142, 0.5178571428571429,
        0.23214285714285715, 0.5178571428571429,
        0.23214285714285715, 0.5178571428571429,
        0.26785714285714285, 0.5178571428571429,
        0.26785714285714285, 0.5178571428571429,
        0.30357142857142855, 0.5178571428571429,
        0.30357142857142855, 0.5178571428571429,
        0.3392857142857143, 0.5178571428571429,
        0.3392857142857143, 0.5178571428571429,
        0.375, 0.5178571428571429,
        0.375, 0.5178571428571429,
        0.4107142857142857, 0.5178571428571429,
        0.4107142857142857, 0.5178571428571429,
        0.44642857142857145, 0.5178571428571429,
        0.44642857142857145, 0.5178571428571429,
        0.48214285714285715, 0.5178571428571429,
        0.48214285714285715, 0.5178571428571429,
        0.5178571428571429, 0.5178571428571429,
        0.5178571428571429, 0.5178571428571429,
        0.5535714285714286, 0.5178571428571429,
        0.5535714285714286, 0.5178571428571429,
        0.5892857142857143, 0.5178571428571429,
        0.5892857142857143, 0.5178571428571429,
        0.625, 0.5178571428571429,
        0.625, 0.5178571428571429,
        0.6607142857142857, 0.5178571428571429,
        0.6607142857142857, 0.5178571428571429,
        0.6964285714285714, 0.5178571428571429,
        0.6964285714285714, 0.5178571428571429,
        0.7321428571428571, 0.5178571428571429,
        0.7321428571428571, 0.5178571428571429,
        0.7678571428571429, 0.5178571428571429,
        0.7678571428571429, 0.5178571428571429,
        0.8035714285714286, 0.5178571428571429,
        0.8035714285714286, 0.5178571428571429,
        0.8392857142857143, 0.5178571428571429,
        0.8392857142857143, 0.5178571428571429,
        0.875, 0.5178571428571429,
        0.875, 0.5178571428571429,
        0.9107142857142857, 0.5178571428571429,
        0.9107142857142857, 0.5178571428571429,
        0.9464285714285714, 0.5178571428571429,
        0.9464285714285714, 0.5178571428571429,
        0.9821428571428571, 0.5178571428571429,
        0.9821428571428571, 0.5178571428571429,
        0.017857142857142856, 0.5535714285714286,
        0.017857142857142856, 0.5535714285714286,
        0.05357142857142857, 0.5535714285714286,
        0.05357142857142857, 0.5535714285714286,
        0.08928571428571429, 0.5535714285714286,
        0.08928571428571429, 0.5535714285714286,
        0.125, 0.5535714285714286,
        0.125, 0.5535714285714286,
        0.16071428571428573, 0.5535714285714286,
        0.16071428571428573, 0.5535714285714286,
        0.19642857142857142, 0.5535714285714286,
        0.19642857142857142, 0.5535714285714286,
        0.23214285714285715, 0.5535714285714286,
        0.23214285714285715, 0.5535714285714286,
        0.26785714285714285, 0.5535714285714286,
        0.26785714285714285, 0.5535714285714286,
        0.30357142857142855, 0.5535714285714286,
        0.30357142857142855, 0.5535714285714286,
        0.3392857142857143, 0.5535714285714286,
        0.3392857142857143, 0.5535714285714286,
        0.375, 0.5535714285714286,
        0.375, 0.5535714285714286,
        0.4107142857142857, 0.5535714285714286,
        0.4107142857142857, 0.5535714285714286,
        0.44642857142857145, 0.5535714285714286,
        0.44642857142857145, 0.5535714285714286,
        0.48214285714285715, 0.5535714285714286,
        0.48214285714285715, 0.5535714285714286,
        0.5178571428571429, 0.5535714285714286,
        0.5178571428571429, 0.5535714285714286,
        0.5535714285714286, 0.5535714285714286,
        0.5535714285714286, 0.5535714285714286,
        0.5892857142857143, 0.5535714285714286,
        0.5892857142857143, 0.5535714285714286,
        0.625, 0.5535714285714286,
        0.625, 0.5535714285714286,
        0.6607142857142857, 0.5535714285714286,
        0.6607142857142857, 0.5535714285714286,
        0.6964285714285714, 0.5535714285714286,
        0.6964285714285714, 0.5535714285714286,
        0.7321428571428571, 0.5535714285714286,
        0.7321428571428571, 0.5535714285714286,
        0.7678571428571429, 0.5535714285714286,
        0.7678571428571429, 0.5535714285714286,
        0.8035714285714286, 0.5535714285714286,
        0.8035714285714286, 0.5535714285714286,
        0.8392857142857143, 0.5535714285714286,
        0.8392857142857143, 0.5535714285714286,
        0.875, 0.5535714285714286,
        0.875, 0.5535714285714286,
        0.9107142857142857, 0.5535714285714286,
        0.9107142857142857, 0.5535714285714286,
        0.9464285714285714, 0.5535714285714286,
        0.9464285714285714, 0.5535714285714286,
        0.9821428571428571, 0.5535714285714286,
        0.9821428571428571, 0.5535714285714286,
        0.017857142857142856, 0.5892857142857143,
        0.017857142857142856, 0.5892857142857143,
        0.05357142857142857, 0.5892857142857143,
        0.05357142857142857, 0.5892857142857143,
        0.08928571428571429, 0.5892857142857143,
        0.08928571428571429, 0.5892857142857143,
        0.125, 0.5892857142857143,
        0.125, 0.5892857142857143,
        0.16071428571428573, 0.5892857142857143,
        0.16071428571428573, 0.5892857142857143,
        0.19642857142857142, 0.5892857142857143,
        0.19642857142857142, 0.5892857142857143,
        0.23214285714285715, 0.5892857142857143,
        0.23214285714285715, 0.5892857142857143,
        0.26785714285714285, 0.5892857142857143,
        0.26785714285714285, 0.5892857142857143,
        0.30357142857142855, 0.5892857142857143,
        0.30357142857142855, 0.5892857142857143,
        0.3392857142857143, 0.5892857142857143,
        0.3392857142857143, 0.5892857142857143,
        0.375, 0.5892857142857143,
        0.375, 0.5892857142857143,
        0.4107142857142857, 0.5892857142857143,
        0.4107142857142857, 0.5892857142857143,
        0.44642857142857145, 0.5892857142857143,
        0.44642857142857145, 0.5892857142857143,
        0.48214285714285715, 0.5892857142857143,
        0.48214285714285715, 0.5892857142857143,
        0.5178571428571429, 0.5892857142857143,
        0.5178571428571429, 0.5892857142857143,
        0.5535714285714286, 0.5892857142857143,
        0.5535714285714286, 0.5892857142857143,
        0.5892857142857143, 0.5892857142857143,
        0.5892857142857143, 0.5892857142857143,
        0.625, 0.5892857142857143,
        0.625, 0.5892857142857143,
        0.6607142857142857, 0.5892857142857143,
        0.6607142857142857, 0.5892857142857143,
        0.6964285714285714, 0.5892857142857143,
        0.6964285714285714, 0.5892857142857143,
        0.7321428571428571, 0.5892857142857143,
        0.7321428571428571, 0.5892857142857143,
        0.7678571428571429, 0.5892857142857143,
        0.7678571428571429, 0.5892857142857143,
        0.8035714285714286, 0.5892857142857143,
        0.8035714285714286, 0.5892857142857143,
        0.8392857142857143, 0.5892857142857143,
        0.8392857142857143, 0.5892857142857143,
        0.875, 0.5892857142857143,
        0.875, 0.5892857142857143,
        0.9107142857142857, 0.5892857142857143,
        0.9107142857142857, 0.5892857142857143,
        0.9464285714285714, 0.5892857142857143,
        0.9464285714285714, 0.5892857142857143,
        0.9821428571428571, 0.5892857142857143,
        0.9821428571428571, 0.5892857142857143,
        0.017857142857142856, 0.625,
        0.017857142857142856, 0.625,
        0.05357142857142857, 0.625,
        0.05357142857142857, 0.625,
        0.08928571428571429, 0.625,
        0.08928571428571429, 0.625,
        0.125, 0.625,
        0.125, 0.625,
        0.16071428571428573, 0.625,
        0.16071428571428573, 0.625,
        0.19642857142857142, 0.625,
        0.19642857142857142, 0.625,
        0.23214285714285715, 0.625,
        0.23214285714285715, 0.625,
        0.26785714285714285, 0.625,
        0.26785714285714285, 0.625,
        0.30357142857142855, 0.625,
        0.30357142857142855, 0.625,
        0.3392857142857143, 0.625,
        0.3392857142857143, 0.625,
        0.375, 0.625,
        0.375, 0.625,
        0.4107142857142857, 0.625,
        0.4107142857142857, 0.625,
        0.44642857142857145, 0.625,
        0.44642857142857145, 0.625,
        0.48214285714285715, 0.625,
        0.48214285714285715, 0.625,
        0.5178571428571429, 0.625,
        0.5178571428571429, 0.625,
        0.5535714285714286, 0.625,
        0.5535714285714286, 0.625,
        0.5892857142857143, 0.625,
        0.5892857142857143, 0.625,
        0.625, 0.625,
        0.625, 0.625,
        0.6607142857142857, 0.625,
        0.6607142857142857, 0.625,
        0.6964285714285714, 0.625,
        0.6964285714285714, 0.625,
        0.7321428571428571, 0.625,
        0.7321428571428571, 0.625,
        0.7678571428571429, 0.625,
        0.7678571428571429, 0.625,
        0.8035714285714286, 0.625,
        0.8035714285714286, 0.625,
        0.8392857142857143, 0.625,
        0.8392857142857143, 0.625,
        0.875, 0.625,
        0.875, 0.625,
        0.9107142857142857, 0.625,
        0.9107142857142857, 0.625,
        0.9464285714285714, 0.625,
        0.9464285714285714, 0.625,
        0.9821428571428571, 0.625,
        0.9821428571428571, 0.625,
        0.017857142857142856, 0.6607142857142857,
        0.017857142857142856, 0.6607142857142857,
        0.05357142857142857, 0.6607142857142857,
        0.05357142857142857, 0.6607142857142857,
        0.08928571428571429, 0.6607142857142857,
        0.08928571428571429, 0.6607142857142857,
        0.125, 0.6607142857142857,
        0.125, 0.6607142857142857,
        0.16071428571428573, 0.6607142857142857,
        0.16071428571428573, 0.6607142857142857,
        0.19642857142857142, 0.6607142857142857,
        0.19642857142857142, 0.6607142857142857,
        0.23214285714285715, 0.6607142857142857,
        0.23214285714285715, 0.6607142857142857,
        0.26785714285714285, 0.6607142857142857,
        0.26785714285714285, 0.6607142857142857,
        0.30357142857142855, 0.6607142857142857,
        0.30357142857142855, 0.6607142857142857,
        0.3392857142857143, 0.6607142857142857,
        0.3392857142857143, 0.6607142857142857,
        0.375, 0.6607142857142857,
        0.375, 0.6607142857142857,
        0.4107142857142857, 0.6607142857142857,
        0.4107142857142857, 0.6607142857142857,
        0.44642857142857145, 0.6607142857142857,
        0.44642857142857145, 0.6607142857142857,
        0.48214285714285715, 0.6607142857142857,
        0.48214285714285715, 0.6607142857142857,
        0.5178571428571429, 0.6607142857142857,
        0.5178571428571429, 0.6607142857142857,
        0.5535714285714286, 0.6607142857142857,
        0.5535714285714286, 0.6607142857142857,
        0.5892857142857143, 0.6607142857142857,
        0.5892857142857143, 0.6607142857142857,
        0.625, 0.6607142857142857,
        0.625, 0.6607142857142857,
        0.6607142857142857, 0.6607142857142857,
        0.6607142857142857, 0.6607142857142857,
        0.6964285714285714, 0.6607142857142857,
        0.6964285714285714, 0.6607142857142857,
        0.7321428571428571, 0.6607142857142857,
        0.7321428571428571, 0.6607142857142857,
        0.7678571428571429, 0.6607142857142857,
        0.7678571428571429, 0.6607142857142857,
        0.8035714285714286, 0.6607142857142857,
        0.8035714285714286, 0.6607142857142857,
        0.8392857142857143, 0.6607142857142857,
        0.8392857142857143, 0.6607142857142857,
        0.875, 0.6607142857142857,
        0.875, 0.6607142857142857,
        0.9107142857142857, 0.6607142857142857,
        0.9107142857142857, 0.6607142857142857,
        0.9464285714285714, 0.6607142857142857,
        0.9464285714285714, 0.6607142857142857,
        0.9821428571428571, 0.6607142857142857,
        0.9821428571428571, 0.6607142857142857,
        0.017857142857142856, 0.6964285714285714,
        0.017857142857142856, 0.6964285714285714,
        0.05357142857142857, 0.6964285714285714,
        0.05357142857142857, 0.6964285714285714,
        0.08928571428571429, 0.6964285714285714,
        0.08928571428571429, 0.6964285714285714,
        0.125, 0.6964285714285714,
        0.125, 0.6964285714285714,
        0.16071428571428573, 0.6964285714285714,
        0.16071428571428573, 0.6964285714285714,
        0.19642857142857142, 0.6964285714285714,
        0.19642857142857142, 0.6964285714285714,
        0.23214285714285715, 0.6964285714285714,
        0.23214285714285715, 0.6964285714285714,
        0.26785714285714285, 0.6964285714285714,
        0.26785714285714285, 0.6964285714285714,
        0.30357142857142855, 0.6964285714285714,
        0.30357142857142855, 0.6964285714285714,
        0.3392857142857143, 0.6964285714285714,
        0.3392857142857143, 0.6964285714285714,
        0.375, 0.6964285714285714,
        0.375, 0.6964285714285714,
        0.4107142857142857, 0.6964285714285714,
        0.4107142857142857, 0.6964285714285714,
        0.44642857142857145, 0.6964285714285714,
        0.44642857142857145, 0.6964285714285714,
        0.48214285714285715, 0.6964285714285714,
        0.48214285714285715, 0.6964285714285714,
        0.5178571428571429, 0.6964285714285714,
        0.5178571428571429, 0.6964285714285714,
        0.5535714285714286, 0.6964285714285714,
        0.5535714285714286, 0.6964285714285714,
        0.5892857142857143, 0.6964285714285714,
        0.5892857142857143, 0.6964285714285714,
        0.625, 0.6964285714285714,
        0.625, 0.6964285714285714,
        0.6607142857142857, 0.6964285714285714,
        0.6607142857142857, 0.6964285714285714,
        0.6964285714285714, 0.6964285714285714,
        0.6964285714285714, 0.6964285714285714,
        0.7321428571428571, 0.6964285714285714,
        0.7321428571428571, 0.6964285714285714,
        0.7678571428571429, 0.6964285714285714,
        0.7678571428571429, 0.6964285714285714,
        0.8035714285714286, 0.6964285714285714,
        0.8035714285714286, 0.6964285714285714,
        0.8392857142857143, 0.6964285714285714,
        0.8392857142857143, 0.6964285714285714,
        0.875, 0.6964285714285714,
        0.875, 0.6964285714285714,
        0.9107142857142857, 0.6964285714285714,
        0.9107142857142857, 0.6964285714285714,
        0.9464285714285714, 0.6964285714285714,
        0.9464285714285714, 0.6964285714285714,
        0.9821428571428571, 0.6964285714285714,
        0.9821428571428571, 0.6964285714285714,
        0.017857142857142856, 0.7321428571428571,
        0.017857142857142856, 0.7321428571428571,
        0.05357142857142857, 0.7321428571428571,
        0.05357142857142857, 0.7321428571428571,
        0.08928571428571429, 0.7321428571428571,
        0.08928571428571429, 0.7321428571428571,
        0.125, 0.7321428571428571,
        0.125, 0.7321428571428571,
        0.16071428571428573, 0.7321428571428571,
        0.16071428571428573, 0.7321428571428571,
        0.19642857142857142, 0.7321428571428571,
        0.19642857142857142, 0.7321428571428571,
        0.23214285714285715, 0.7321428571428571,
        0.23214285714285715, 0.7321428571428571,
        0.26785714285714285, 0.7321428571428571,
        0.26785714285714285, 0.7321428571428571,
        0.30357142857142855, 0.7321428571428571,
        0.30357142857142855, 0.7321428571428571,
        0.3392857142857143, 0.7321428571428571,
        0.3392857142857143, 0.7321428571428571,
        0.375, 0.7321428571428571,
        0.375, 0.7321428571428571,
        0.4107142857142857, 0.7321428571428571,
        0.4107142857142857, 0.7321428571428571,
        0.44642857142857145, 0.7321428571428571,
        0.44642857142857145, 0.7321428571428571,
        0.48214285714285715, 0.7321428571428571,
        0.48214285714285715, 0.7321428571428571,
        0.5178571428571429, 0.7321428571428571,
        0.5178571428571429, 0.7321428571428571,
        0.5535714285714286, 0.7321428571428571,
        0.5535714285714286, 0.7321428571428571,
        0.5892857142857143, 0.7321428571428571,
        0.5892857142857143, 0.7321428571428571,
        0.625, 0.7321428571428571,
        0.625, 0.7321428571428571,
        0.6607142857142857, 0.7321428571428571,
        0.6607142857142857, 0.7321428571428571,
        0.6964285714285714, 0.7321428571428571,
        0.6964285714285714, 0.7321428571428571,
        0.7321428571428571, 0.7321428571428571,
        0.7321428571428571, 0.7321428571428571,
        0.7678571428571429, 0.7321428571428571,
        0.7678571428571429, 0.7321428571428571,
        0.8035714285714286, 0.7321428571428571,
        0.8035714285714286, 0.7321428571428571,
        0.8392857142857143, 0.7321428571428571,
        0.8392857142857143, 0.7321428571428571,
        0.875, 0.7321428571428571,
        0.875, 0.7321428571428571,
        0.9107142857142857, 0.7321428571428571,
        0.9107142857142857, 0.7321428571428571,
        0.9464285714285714, 0.7321428571428571,
        0.9464285714285714, 0.7321428571428571,
        0.9821428571428571, 0.7321428571428571,
        0.9821428571428571, 0.7321428571428571,
        0.017857142857142856, 0.7678571428571429,
        0.017857142857142856, 0.7678571428571429,
        0.05357142857142857, 0.7678571428571429,
        0.05357142857142857, 0.7678571428571429,
        0.08928571428571429, 0.7678571428571429,
        0.08928571428571429, 0.7678571428571429,
        0.125, 0.7678571428571429,
        0.125, 0.7678571428571429,
        0.16071428571428573, 0.7678571428571429,
        0.16071428571428573, 0.7678571428571429,
        0.19642857142857142, 0.7678571428571429,
        0.19642857142857142, 0.7678571428571429,
        0.23214285714285715, 0.7678571428571429,
        0.23214285714285715, 0.7678571428571429,
        0.26785714285714285, 0.7678571428571429,
        0.26785714285714285, 0.7678571428571429,
        0.30357142857142855, 0.7678571428571429,
        0.30357142857142855, 0.7678571428571429,
        0.3392857142857143, 0.7678571428571429,
        0.3392857142857143, 0.7678571428571429,
        0.375, 0.7678571428571429,
        0.375, 0.7678571428571429,
        0.4107142857142857, 0.7678571428571429,
        0.4107142857142857, 0.7678571428571429,
        0.44642857142857145, 0.7678571428571429,
        0.44642857142857145, 0.7678571428571429,
        0.48214285714285715, 0.7678571428571429,
        0.48214285714285715, 0.7678571428571429,
        0.5178571428571429, 0.7678571428571429,
        0.5178571428571429, 0.7678571428571429,
        0.5535714285714286, 0.7678571428571429,
        0.5535714285714286, 0.7678571428571429,
        0.5892857142857143, 0.7678571428571429,
        0.5892857142857143, 0.7678571428571429,
        0.625, 0.7678571428571429,
        0.625, 0.7678571428571429,
        0.6607142857142857, 0.7678571428571429,
        0.6607142857142857, 0.7678571428571429,
        0.6964285714285714, 0.7678571428571429,
        0.6964285714285714, 0.7678571428571429,
        0.7321428571428571, 0.7678571428571429,
        0.7321428571428571, 0.7678571428571429,
        0.7678571428571429, 0.7678571428571429,
        0.7678571428571429, 0.7678571428571429,
        0.8035714285714286, 0.7678571428571429,
        0.8035714285714286, 0.7678571428571429,
        0.8392857142857143, 0.7678571428571429,
        0.8392857142857143, 0.7678571428571429,
        0.875, 0.7678571428571429,
        0.875, 0.7678571428571429,
        0.9107142857142857, 0.7678571428571429,
        0.9107142857142857, 0.7678571428571429,
        0.9464285714285714, 0.7678571428571429,
        0.9464285714285714, 0.7678571428571429,
        0.9821428571428571, 0.7678571428571429,
        0.9821428571428571, 0.7678571428571429,
        0.017857142857142856, 0.8035714285714286,
        0.017857142857142856, 0.8035714285714286,
        0.05357142857142857, 0.8035714285714286,
        0.05357142857142857, 0.8035714285714286,
        0.08928571428571429, 0.8035714285714286,
        0.08928571428571429, 0.8035714285714286,
        0.125, 0.8035714285714286,
        0.125, 0.8035714285714286,
        0.16071428571428573, 0.8035714285714286,
        0.16071428571428573, 0.8035714285714286,
        0.19642857142857142, 0.8035714285714286,
        0.19642857142857142, 0.8035714285714286,
        0.23214285714285715, 0.8035714285714286,
        0.23214285714285715, 0.8035714285714286,
        0.26785714285714285, 0.8035714285714286,
        0.26785714285714285, 0.8035714285714286,
        0.30357142857142855, 0.8035714285714286,
        0.30357142857142855, 0.8035714285714286,
        0.3392857142857143, 0.8035714285714286,
        0.3392857142857143, 0.8035714285714286,
        0.375, 0.8035714285714286,
        0.375, 0.8035714285714286,
        0.4107142857142857, 0.8035714285714286,
        0.4107142857142857, 0.8035714285714286,
        0.44642857142857145, 0.8035714285714286,
        0.44642857142857145, 0.8035714285714286,
        0.48214285714285715, 0.8035714285714286,
        0.48214285714285715, 0.8035714285714286,
        0.5178571428571429, 0.8035714285714286,
        0.5178571428571429, 0.8035714285714286,
        0.5535714285714286, 0.8035714285714286,
        0.5535714285714286, 0.8035714285714286,
        0.5892857142857143, 0.8035714285714286,
        0.5892857142857143, 0.8035714285714286,
        0.625, 0.8035714285714286,
        0.625, 0.8035714285714286,
        0.6607142857142857, 0.8035714285714286,
        0.6607142857142857, 0.8035714285714286,
        0.6964285714285714, 0.8035714285714286,
        0.6964285714285714, 0.8035714285714286,
        0.7321428571428571, 0.8035714285714286,
        0.7321428571428571, 0.8035714285714286,
        0.7678571428571429, 0.8035714285714286,
        0.7678571428571429, 0.8035714285714286,
        0.8035714285714286, 0.8035714285714286,
        0.8035714285714286, 0.8035714285714286,
        0.8392857142857143, 0.8035714285714286,
        0.8392857142857143, 0.8035714285714286,
        0.875, 0.8035714285714286,
        0.875, 0.8035714285714286,
        0.9107142857142857, 0.8035714285714286,
        0.9107142857142857, 0.8035714285714286,
        0.9464285714285714, 0.8035714285714286,
        0.9464285714285714, 0.8035714285714286,
        0.9821428571428571, 0.8035714285714286,
        0.9821428571428571, 0.8035714285714286,
        0.017857142857142856, 0.8392857142857143,
        0.017857142857142856, 0.8392857142857143,
        0.05357142857142857, 0.8392857142857143,
        0.05357142857142857, 0.8392857142857143,
        0.08928571428571429, 0.8392857142857143,
        0.08928571428571429, 0.8392857142857143,
        0.125, 0.8392857142857143,
        0.125, 0.8392857142857143,
        0.16071428571428573, 0.8392857142857143,
        0.16071428571428573, 0.8392857142857143,
        0.19642857142857142, 0.8392857142857143,
        0.19642857142857142, 0.8392857142857143,
        0.23214285714285715, 0.8392857142857143,
        0.23214285714285715, 0.8392857142857143,
        0.26785714285714285, 0.8392857142857143,
        0.26785714285714285, 0.8392857142857143,
        0.30357142857142855, 0.8392857142857143,
        0.30357142857142855, 0.8392857142857143,
        0.3392857142857143, 0.8392857142857143,
        0.3392857142857143, 0.8392857142857143,
        0.375, 0.8392857142857143,
        0.375, 0.8392857142857143,
        0.4107142857142857, 0.8392857142857143,
        0.4107142857142857, 0.8392857142857143,
        0.44642857142857145, 0.8392857142857143,
        0.44642857142857145, 0.8392857142857143,
        0.48214285714285715, 0.8392857142857143,
        0.48214285714285715, 0.8392857142857143,
        0.5178571428571429, 0.8392857142857143,
        0.5178571428571429, 0.8392857142857143,
        0.5535714285714286, 0.8392857142857143,
        0.5535714285714286, 0.8392857142857143,
        0.5892857142857143, 0.8392857142857143,
        0.5892857142857143, 0.8392857142857143,
        0.625, 0.8392857142857143,
        0.625, 0.8392857142857143,
        0.6607142857142857, 0.8392857142857143,
        0.6607142857142857, 0.8392857142857143,
        0.6964285714285714, 0.8392857142857143,
        0.6964285714285714, 0.8392857142857143,
        0.7321428571428571, 0.8392857142857143,
        0.7321428571428571, 0.8392857142857143,
        0.7678571428571429, 0.8392857142857143,
        0.7678571428571429, 0.8392857142857143,
        0.8035714285714286, 0.8392857142857143,
        0.8035714285714286, 0.8392857142857143,
        0.8392857142857143, 0.8392857142857143,
        0.8392857142857143, 0.8392857142857143,
        0.875, 0.8392857142857143,
        0.875, 0.8392857142857143,
        0.9107142857142857, 0.8392857142857143,
        0.9107142857142857, 0.8392857142857143,
        0.9464285714285714, 0.8392857142857143,
        0.9464285714285714, 0.8392857142857143,
        0.9821428571428571, 0.8392857142857143,
        0.9821428571428571, 0.8392857142857143,
        0.017857142857142856, 0.875,
        0.017857142857142856, 0.875,
        0.05357142857142857, 0.875,
        0.05357142857142857, 0.875,
        0.08928571428571429, 0.875,
        0.08928571428571429, 0.875,
        0.125, 0.875,
        0.125, 0.875,
        0.16071428571428573, 0.875,
        0.16071428571428573, 0.875,
        0.19642857142857142, 0.875,
        0.19642857142857142, 0.875,
        0.23214285714285715, 0.875,
        0.23214285714285715, 0.875,
        0.26785714285714285, 0.875,
        0.26785714285714285, 0.875,
        0.30357142857142855, 0.875,
        0.30357142857142855, 0.875,
        0.3392857142857143, 0.875,
        0.3392857142857143, 0.875,
        0.375, 0.875,
        0.375, 0.875,
        0.4107142857142857, 0.875,
        0.4107142857142857, 0.875,
        0.44642857142857145, 0.875,
        0.44642857142857145, 0.875,
        0.48214285714285715, 0.875,
        0.48214285714285715, 0.875,
        0.5178571428571429, 0.875,
        0.5178571428571429, 0.875,
        0.5535714285714286, 0.875,
        0.5535714285714286, 0.875,
        0.5892857142857143, 0.875,
        0.5892857142857143, 0.875,
        0.625, 0.875,
        0.625, 0.875,
        0.6607142857142857, 0.875,
        0.6607142857142857, 0.875,
        0.6964285714285714, 0.875,
        0.6964285714285714, 0.875,
        0.7321428571428571, 0.875,
        0.7321428571428571, 0.875,
        0.7678571428571429, 0.875,
        0.7678571428571429, 0.875,
        0.8035714285714286, 0.875,
        0.8035714285714286, 0.875,
        0.8392857142857143, 0.875,
        0.8392857142857143, 0.875,
        0.875, 0.875,
        0.875, 0.875,
        0.9107142857142857, 0.875,
        0.9107142857142857, 0.875,
        0.9464285714285714, 0.875,
        0.9464285714285714, 0.875,
        0.9821428571428571, 0.875,
        0.9821428571428571, 0.875,
        0.017857142857142856, 0.9107142857142857,
        0.017857142857142856, 0.9107142857142857,
        0.05357142857142857, 0.9107142857142857,
        0.05357142857142857, 0.9107142857142857,
        0.08928571428571429, 0.9107142857142857,
        0.08928571428571429, 0.9107142857142857,
        0.125, 0.9107142857142857,
        0.125, 0.9107142857142857,
        0.16071428571428573, 0.9107142857142857,
        0.16071428571428573, 0.9107142857142857,
        0.19642857142857142, 0.9107142857142857,
        0.19642857142857142, 0.9107142857142857,
        0.23214285714285715, 0.9107142857142857,
        0.23214285714285715, 0.9107142857142857,
        0.26785714285714285, 0.9107142857142857,
        0.26785714285714285, 0.9107142857142857,
        0.30357142857142855, 0.9107142857142857,
        0.30357142857142855, 0.9107142857142857,
        0.3392857142857143, 0.9107142857142857,
        0.3392857142857143, 0.9107142857142857,
        0.375, 0.9107142857142857,
        0.375, 0.9107142857142857,
        0.4107142857142857, 0.9107142857142857,
        0.4107142857142857, 0.9107142857142857,
        0.44642857142857145, 0.9107142857142857,
        0.44642857142857145, 0.9107142857142857,
        0.48214285714285715, 0.9107142857142857,
        0.48214285714285715, 0.9107142857142857,
        0.5178571428571429, 0.9107142857142857,
        0.5178571428571429, 0.9107142857142857,
        0.5535714285714286, 0.9107142857142857,
        0.5535714285714286, 0.9107142857142857,
        0.5892857142857143, 0.9107142857142857,
        0.5892857142857143, 0.9107142857142857,
        0.625, 0.9107142857142857,
        0.625, 0.9107142857142857,
        0.6607142857142857, 0.9107142857142857,
        0.6607142857142857, 0.9107142857142857,
        0.6964285714285714, 0.9107142857142857,
        0.6964285714285714, 0.9107142857142857,
        0.7321428571428571, 0.9107142857142857,
        0.7321428571428571, 0.9107142857142857,
        0.7678571428571429, 0.9107142857142857,
        0.7678571428571429, 0.9107142857142857,
        0.8035714285714286, 0.9107142857142857,
        0.8035714285714286, 0.9107142857142857,
        0.8392857142857143, 0.9107142857142857,
        0.8392857142857143, 0.9107142857142857,
        0.875, 0.9107142857142857,
        0.875, 0.9107142857142857,
        0.9107142857142857, 0.9107142857142857,
        0.9107142857142857, 0.9107142857142857,
        0.9464285714285714, 0.9107142857142857,
        0.9464285714285714, 0.9107142857142857,
        0.9821428571428571, 0.9107142857142857,
        0.9821428571428571, 0.9107142857142857,
        0.017857142857142856, 0.9464285714285714,
        0.017857142857142856, 0.9464285714285714,
        0.05357142857142857, 0.9464285714285714,
        0.05357142857142857, 0.9464285714285714,
        0.08928571428571429, 0.9464285714285714,
        0.08928571428571429, 0.9464285714285714,
        0.125, 0.9464285714285714,
        0.125, 0.9464285714285714,
        0.16071428571428573, 0.9464285714285714,
        0.16071428571428573, 0.9464285714285714,
        0.19642857142857142, 0.9464285714285714,
        0.19642857142857142, 0.9464285714285714,
        0.23214285714285715, 0.9464285714285714,
        0.23214285714285715, 0.9464285714285714,
        0.26785714285714285, 0.9464285714285714,
        0.26785714285714285, 0.9464285714285714,
        0.30357142857142855, 0.9464285714285714,
        0.30357142857142855, 0.9464285714285714,
        0.3392857142857143, 0.9464285714285714,
        0.3392857142857143, 0.9464285714285714,
        0.375, 0.9464285714285714,
        0.375, 0.9464285714285714,
        0.4107142857142857, 0.9464285714285714,
        0.4107142857142857, 0.9464285714285714,
        0.44642857142857145, 0.9464285714285714,
        0.44642857142857145, 0.9464285714285714,
        0.48214285714285715, 0.9464285714285714,
        0.48214285714285715, 0.9464285714285714,
        0.5178571428571429, 0.9464285714285714,
        0.5178571428571429, 0.9464285714285714,
        0.5535714285714286, 0.9464285714285714,
        0.5535714285714286, 0.9464285714285714,
        0.5892857142857143, 0.9464285714285714,
        0.5892857142857143, 0.9464285714285714,
        0.625, 0.9464285714285714,
        0.625, 0.9464285714285714,
        0.6607142857142857, 0.9464285714285714,
        0.6607142857142857, 0.9464285714285714,
        0.6964285714285714, 0.9464285714285714,
        0.6964285714285714, 0.9464285714285714,
        0.7321428571428571, 0.9464285714285714,
        0.7321428571428571, 0.9464285714285714,
        0.7678571428571429, 0.9464285714285714,
        0.7678571428571429, 0.9464285714285714,
        0.8035714285714286, 0.9464285714285714,
        0.8035714285714286, 0.9464285714285714,
        0.8392857142857143, 0.9464285714285714,
        0.8392857142857143, 0.9464285714285714,
        0.875, 0.9464285714285714,
        0.875, 0.9464285714285714,
        0.9107142857142857, 0.9464285714285714,
        0.9107142857142857, 0.9464285714285714,
        0.9464285714285714, 0.9464285714285714,
        0.9464285714285714, 0.9464285714285714,
        0.9821428571428571, 0.9464285714285714,
        0.9821428571428571, 0.9464285714285714,
        0.017857142857142856, 0.9821428571428571,
        0.017857142857142856, 0.9821428571428571,
        0.05357142857142857, 0.9821428571428571,
        0.05357142857142857, 0.9821428571428571,
        0.08928571428571429, 0.9821428571428571,
        0.08928571428571429, 0.9821428571428571,
        0.125, 0.9821428571428571,
        0.125, 0.9821428571428571,
        0.16071428571428573, 0.9821428571428571,
        0.16071428571428573, 0.9821428571428571,
        0.19642857142857142, 0.9821428571428571,
        0.19642857142857142, 0.9821428571428571,
        0.23214285714285715, 0.9821428571428571,
        0.23214285714285715, 0.9821428571428571,
        0.26785714285714285, 0.9821428571428571,
        0.26785714285714285, 0.9821428571428571,
        0.30357142857142855, 0.9821428571428571,
        0.30357142857142855, 0.9821428571428571,
        0.3392857142857143, 0.9821428571428571,
        0.3392857142857143, 0.9821428571428571,
        0.375, 0.9821428571428571,
        0.375, 0.9821428571428571,
        0.4107142857142857, 0.9821428571428571,
        0.4107142857142857, 0.9821428571428571,
        0.44642857142857145, 0.9821428571428571,
        0.44642857142857145, 0.9821428571428571,
        0.48214285714285715, 0.9821428571428571,
        0.48214285714285715, 0.9821428571428571,
        0.5178571428571429, 0.9821428571428571,
        0.5178571428571429, 0.9821428571428571,
        0.5535714285714286, 0.9821428571428571,
        0.5535714285714286, 0.9821428571428571,
        0.5892857142857143, 0.9821428571428571,
        0.5892857142857143, 0.9821428571428571,
        0.625, 0.9821428571428571,
        0.625, 0.9821428571428571,
        0.6607142857142857, 0.9821428571428571,
        0.6607142857142857, 0.9821428571428571,
        0.6964285714285714, 0.9821428571428571,
        0.6964285714285714, 0.9821428571428571,
        0.7321428571428571, 0.9821428571428571,
        0.7321428571428571, 0.9821428571428571,
        0.7678571428571429, 0.9821428571428571,
        0.7678571428571429, 0.9821428571428571,
        0.8035714285714286, 0.9821428571428571,
        0.8035714285714286, 0.9821428571428571,
        0.8392857142857143, 0.9821428571428571,
        0.8392857142857143, 0.9821428571428571,
        0.875, 0.9821428571428571,
        0.875, 0.9821428571428571,
        0.9107142857142857, 0.9821428571428571,
        0.9107142857142857, 0.9821428571428571,
        0.9464285714285714, 0.9821428571428571,
        0.9464285714285714, 0.9821428571428571,
        0.9821428571428571, 0.9821428571428571,
        0.9821428571428571, 0.9821428571428571,
        0.03571428571428571, 0.03571428571428571,
        0.03571428571428571, 0.03571428571428571,
        0.10714285714285714, 0.03571428571428571,
        0.10714285714285714, 0.03571428571428571,
        0.17857142857142858, 0.03571428571428571,
        0.17857142857142858, 0.03571428571428571,
        0.25, 0.03571428571428571,
        0.25, 0.03571428571428571,
        0.32142857142857145, 0.03571428571428571,
        0.32142857142857145, 0.03571428571428571,
        0.39285714285714285, 0.03571428571428571,
        0.39285714285714285, 0.03571428571428571,
        0.4642857142857143, 0.03571428571428571,
        0.4642857142857143, 0.03571428571428571,
        0.5357142857142857, 0.03571428571428571,
        0.5357142857142857, 0.03571428571428571,
        0.6071428571428571, 0.03571428571428571,
        0.6071428571428571, 0.03571428571428571,
        0.6785714285714286, 0.03571428571428571,
        0.6785714285714286, 0.03571428571428571,
        0.75, 0.03571428571428571,
        0.75, 0.03571428571428571,
        0.8214285714285714, 0.03571428571428571,
        0.8214285714285714, 0.03571428571428571,
        0.8928571428571429, 0.03571428571428571,
        0.8928571428571429, 0.03571428571428571,
        0.9642857142857143, 0.03571428571428571,
        0.9642857142857143, 0.03571428571428571,
        0.03571428571428571, 0.10714285714285714,
        0.03571428571428571, 0.10714285714285714,
        0.10714285714285714, 0.10714285714285714,
        0.10714285714285714, 0.10714285714285714,
        0.17857142857142858, 0.10714285714285714,
        0.17857142857142858, 0.10714285714285714,
        0.25, 0.10714285714285714,
        0.25, 0.10714285714285714,
        0.32142857142857145, 0.10714285714285714,
        0.32142857142857145, 0.10714285714285714,
        0.39285714285714285, 0.10714285714285714,
        0.39285714285714285, 0.10714285714285714,
        0.4642857142857143, 0.10714285714285714,
        0.4642857142857143, 0.10714285714285714,
        0.5357142857142857, 0.10714285714285714,
        0.5357142857142857, 0.10714285714285714,
        0.6071428571428571, 0.10714285714285714,
        0.6071428571428571, 0.10714285714285714,
        0.6785714285714286, 0.10714285714285714,
        0.6785714285714286, 0.10714285714285714,
        0.75, 0.10714285714285714,
        0.75, 0.10714285714285714,
        0.8214285714285714, 0.10714285714285714,
        0.8214285714285714, 0.10714285714285714,
        0.8928571428571429, 0.10714285714285714,
        0.8928571428571429, 0.10714285714285714,
        0.9642857142857143, 0.10714285714285714,
        0.9642857142857143, 0.10714285714285714,
        0.03571428571428571, 0.17857142857142858,
        0.03571428571428571, 0.17857142857142858,
        0.10714285714285714, 0.17857142857142858,
        0.10714285714285714, 0.17857142857142858,
        0.17857142857142858, 0.17857142857142858,
        0.17857142857142858, 0.17857142857142858,
        0.25, 0.17857142857142858,
        0.25, 0.17857142857142858,
        0.32142857142857145, 0.17857142857142858,
        0.32142857142857145, 0.17857142857142858,
        0.39285714285714285, 0.17857142857142858,
        0.39285714285714285, 0.17857142857142858,
        0.4642857142857143, 0.17857142857142858,
        0.4642857142857143, 0.17857142857142858,
        0.5357142857142857, 0.17857142857142858,
        0.5357142857142857, 0.17857142857142858,
        0.6071428571428571, 0.17857142857142858,
        0.6071428571428571, 0.17857142857142858,
        0.6785714285714286, 0.17857142857142858,
        0.6785714285714286, 0.17857142857142858,
        0.75, 0.17857142857142858,
        0.75, 0.17857142857142858,
        0.8214285714285714, 0.17857142857142858,
        0.8214285714285714, 0.17857142857142858,
        0.8928571428571429, 0.17857142857142858,
        0.8928571428571429, 0.17857142857142858,
        0.9642857142857143, 0.17857142857142858,
        0.9642857142857143, 0.17857142857142858,
        0.03571428571428571, 0.25,
        0.03571428571428571, 0.25,
        0.10714285714285714, 0.25,
        0.10714285714285714, 0.25,
        0.17857142857142858, 0.25,
        0.17857142857142858, 0.25,
        0.25, 0.25,
        0.25, 0.25,
        0.32142857142857145, 0.25,
        0.32142857142857145, 0.25,
        0.39285714285714285, 0.25,
        0.39285714285714285, 0.25,
        0.4642857142857143, 0.25,
        0.4642857142857143, 0.25,
        0.5357142857142857, 0.25,
        0.5357142857142857, 0.25,
        0.6071428571428571, 0.25,
        0.6071428571428571, 0.25,
        0.6785714285714286, 0.25,
        0.6785714285714286, 0.25,
        0.75, 0.25,
        0.75, 0.25,
        0.8214285714285714, 0.25,
        0.8214285714285714, 0.25,
        0.8928571428571429, 0.25,
        0.8928571428571429, 0.25,
        0.9642857142857143, 0.25,
        0.9642857142857143, 0.25,
        0.03571428571428571, 0.32142857142857145,
        0.03571428571428571, 0.32142857142857145,
        0.10714285714285714, 0.32142857142857145,
        0.10714285714285714, 0.32142857142857145,
        0.17857142857142858, 0.32142857142857145,
        0.17857142857142858, 0.32142857142857145,
        0.25, 0.32142857142857145,
        0.25, 0.32142857142857145,
        0.32142857142857145, 0.32142857142857145,
        0.32142857142857145, 0.32142857142857145,
        0.39285714285714285, 0.32142857142857145,
        0.39285714285714285, 0.32142857142857145,
        0.4642857142857143, 0.32142857142857145,
        0.4642857142857143, 0.32142857142857145,
        0.5357142857142857, 0.32142857142857145,
        0.5357142857142857, 0.32142857142857145,
        0.6071428571428571, 0.32142857142857145,
        0.6071428571428571, 0.32142857142857145,
        0.6785714285714286, 0.32142857142857145,
        0.6785714285714286, 0.32142857142857145,
        0.75, 0.32142857142857145,
        0.75, 0.32142857142857145,
        0.8214285714285714, 0.32142857142857145,
        0.8214285714285714, 0.32142857142857145,
        0.8928571428571429, 0.32142857142857145,
        0.8928571428571429, 0.32142857142857145,
        0.9642857142857143, 0.32142857142857145,
        0.9642857142857143, 0.32142857142857145,
        0.03571428571428571, 0.39285714285714285,
        0.03571428571428571, 0.39285714285714285,
        0.10714285714285714, 0.39285714285714285,
        0.10714285714285714, 0.39285714285714285,
        0.17857142857142858, 0.39285714285714285,
        0.17857142857142858, 0.39285714285714285,
        0.25, 0.39285714285714285,
        0.25, 0.39285714285714285,
        0.32142857142857145, 0.39285714285714285,
        0.32142857142857145, 0.39285714285714285,
        0.39285714285714285, 0.39285714285714285,
        0.39285714285714285, 0.39285714285714285,
        0.4642857142857143, 0.39285714285714285,
        0.4642857142857143, 0.39285714285714285,
        0.5357142857142857, 0.39285714285714285,
        0.5357142857142857, 0.39285714285714285,
        0.6071428571428571, 0.39285714285714285,
        0.6071428571428571, 0.39285714285714285,
        0.6785714285714286, 0.39285714285714285,
        0.6785714285714286, 0.39285714285714285,
        0.75, 0.39285714285714285,
        0.75, 0.39285714285714285,
        0.8214285714285714, 0.39285714285714285,
        0.8214285714285714, 0.39285714285714285,
        0.8928571428571429, 0.39285714285714285,
        0.8928571428571429, 0.39285714285714285,
        0.9642857142857143, 0.39285714285714285,
        0.9642857142857143, 0.39285714285714285,
        0.03571428571428571, 0.4642857142857143,
        0.03571428571428571, 0.4642857142857143,
        0.10714285714285714, 0.4642857142857143,
        0.10714285714285714, 0.4642857142857143,
        0.17857142857142858, 0.4642857142857143,
        0.17857142857142858, 0.4642857142857143,
        0.25, 0.4642857142857143,
        0.25, 0.4642857142857143,
        0.32142857142857145, 0.4642857142857143,
        0.32142857142857145, 0.4642857142857143,
        0.39285714285714285, 0.4642857142857143,
        0.39285714285714285, 0.4642857142857143,
        0.4642857142857143, 0.4642857142857143,
        0.4642857142857143, 0.4642857142857143,
        0.5357142857142857, 0.4642857142857143,
        0.5357142857142857, 0.4642857142857143,
        0.6071428571428571, 0.4642857142857143,
        0.6071428571428571, 0.4642857142857143,
        0.6785714285714286, 0.4642857142857143,
        0.6785714285714286, 0.4642857142857143,
        0.75, 0.4642857142857143,
        0.75, 0.4642857142857143,
        0.8214285714285714, 0.4642857142857143,
        0.8214285714285714, 0.4642857142857143,
        0.8928571428571429, 0.4642857142857143,
        0.8928571428571429, 0.4642857142857143,
        0.9642857142857143, 0.4642857142857143,
        0.9642857142857143, 0.4642857142857143,
        0.03571428571428571, 0.5357142857142857,
        0.03571428571428571, 0.5357142857142857,
        0.10714285714285714, 0.5357142857142857,
        0.10714285714285714, 0.5357142857142857,
        0.17857142857142858, 0.5357142857142857,
        0.17857142857142858, 0.5357142857142857,
        0.25, 0.5357142857142857,
        0.25, 0.5357142857142857,
        0.32142857142857145, 0.5357142857142857,
        0.32142857142857145, 0.5357142857142857,
        0.39285714285714285, 0.5357142857142857,
        0.39285714285714285, 0.5357142857142857,
        0.4642857142857143, 0.5357142857142857,
        0.4642857142857143, 0.5357142857142857,
        0.5357142857142857, 0.5357142857142857,
        0.5357142857142857, 0.5357142857142857,
        0.6071428571428571, 0.5357142857142857,
        0.6071428571428571, 0.5357142857142857,
        0.6785714285714286, 0.5357142857142857,
        0.6785714285714286, 0.5357142857142857,
        0.75, 0.5357142857142857,
        0.75, 0.5357142857142857,
        0.8214285714285714, 0.5357142857142857,
        0.8214285714285714, 0.5357142857142857,
        0.8928571428571429, 0.5357142857142857,
        0.8928571428571429, 0.5357142857142857,
        0.9642857142857143, 0.5357142857142857,
        0.9642857142857143, 0.5357142857142857,
        0.03571428571428571, 0.6071428571428571,
        0.03571428571428571, 0.6071428571428571,
        0.10714285714285714, 0.6071428571428571,
        0.10714285714285714, 0.6071428571428571,
        0.17857142857142858, 0.6071428571428571,
        0.17857142857142858, 0.6071428571428571,
        0.25, 0.6071428571428571,
        0.25, 0.6071428571428571,
        0.32142857142857145, 0.6071428571428571,
        0.32142857142857145, 0.6071428571428571,
        0.39285714285714285, 0.6071428571428571,
        0.39285714285714285, 0.6071428571428571,
        0.4642857142857143, 0.6071428571428571,
        0.4642857142857143, 0.6071428571428571,
        0.5357142857142857, 0.6071428571428571,
        0.5357142857142857, 0.6071428571428571,
        0.6071428571428571, 0.6071428571428571,
        0.6071428571428571, 0.6071428571428571,
        0.6785714285714286, 0.6071428571428571,
        0.6785714285714286, 0.6071428571428571,
        0.75, 0.6071428571428571,
        0.75, 0.6071428571428571,
        0.8214285714285714, 0.6071428571428571,
        0.8214285714285714, 0.6071428571428571,
        0.8928571428571429, 0.6071428571428571,
        0.8928571428571429, 0.6071428571428571,
        0.9642857142857143, 0.6071428571428571,
        0.9642857142857143, 0.6071428571428571,
        0.03571428571428571, 0.6785714285714286,
        0.03571428571428571, 0.6785714285714286,
        0.10714285714285714, 0.6785714285714286,
        0.10714285714285714, 0.6785714285714286,
        0.17857142857142858, 0.6785714285714286,
        0.17857142857142858, 0.6785714285714286,
        0.25, 0.6785714285714286,
        0.25, 0.6785714285714286,
        0.32142857142857145, 0.6785714285714286,
        0.32142857142857145, 0.6785714285714286,
        0.39285714285714285, 0.6785714285714286,
        0.39285714285714285, 0.6785714285714286,
        0.4642857142857143, 0.6785714285714286,
        0.4642857142857143, 0.6785714285714286,
        0.5357142857142857, 0.6785714285714286,
        0.5357142857142857, 0.6785714285714286,
        0.6071428571428571, 0.6785714285714286,
        0.6071428571428571, 0.6785714285714286,
        0.6785714285714286, 0.6785714285714286,
        0.6785714285714286, 0.6785714285714286,
        0.75, 0.6785714285714286,
        0.75, 0.6785714285714286,
        0.8214285714285714, 0.6785714285714286,
        0.8214285714285714, 0.6785714285714286,
        0.8928571428571429, 0.6785714285714286,
        0.8928571428571429, 0.6785714285714286,
        0.9642857142857143, 0.6785714285714286,
        0.9642857142857143, 0.6785714285714286,
        0.03571428571428571, 0.75,
        0.03571428571428571, 0.75,
        0.10714285714285714, 0.75,
        0.10714285714285714, 0.75,
        0.17857142857142858, 0.75,
        0.17857142857142858, 0.75,
        0.25, 0.75,
        0.25, 0.75,
        0.32142857142857145, 0.75,
        0.32142857142857145, 0.75,
        0.39285714285714285, 0.75,
        0.39285714285714285, 0.75,
        0.4642857142857143, 0.75,
        0.4642857142857143, 0.75,
        0.5357142857142857, 0.75,
        0.5357142857142857, 0.75,
        0.6071428571428571, 0.75,
        0.6071428571428571, 0.75,
        0.6785714285714286, 0.75,
        0.6785714285714286, 0.75,
        0.75, 0.75,
        0.75, 0.75,
        0.8214285714285714, 0.75,
        0.8214285714285714, 0.75,
        0.8928571428571429, 0.75,
        0.8928571428571429, 0.75,
        0.9642857142857143, 0.75,
        0.9642857142857143, 0.75,
        0.03571428571428571, 0.8214285714285714,
        0.03571428571428571, 0.8214285714285714,
        0.10714285714285714, 0.8214285714285714,
        0.10714285714285714, 0.8214285714285714,
        0.17857142857142858, 0.8214285714285714,
        0.17857142857142858, 0.8214285714285714,
        0.25, 0.8214285714285714,
        0.25, 0.8214285714285714,
        0.32142857142857145, 0.8214285714285714,
        0.32142857142857145, 0.8214285714285714,
        0.39285714285714285, 0.8214285714285714,
        0.39285714285714285, 0.8214285714285714,
        0.4642857142857143, 0.8214285714285714,
        0.4642857142857143, 0.8214285714285714,
        0.5357142857142857, 0.8214285714285714,
        0.5357142857142857, 0.8214285714285714,
        0.6071428571428571, 0.8214285714285714,
        0.6071428571428571, 0.8214285714285714,
        0.6785714285714286, 0.8214285714285714,
        0.6785714285714286, 0.8214285714285714,
        0.75, 0.8214285714285714,
        0.75, 0.8214285714285714,
        0.8214285714285714, 0.8214285714285714,
        0.8214285714285714, 0.8214285714285714,
        0.8928571428571429, 0.8214285714285714,
        0.8928571428571429, 0.8214285714285714,
        0.9642857142857143, 0.8214285714285714,
        0.9642857142857143, 0.8214285714285714,
        0.03571428571428571, 0.8928571428571429,
        0.03571428571428571, 0.8928571428571429,
        0.10714285714285714, 0.8928571428571429,
        0.10714285714285714, 0.8928571428571429,
        0.17857142857142858, 0.8928571428571429,
        0.17857142857142858, 0.8928571428571429,
        0.25, 0.8928571428571429,
        0.25, 0.8928571428571429,
        0.32142857142857145, 0.8928571428571429,
        0.32142857142857145, 0.8928571428571429,
        0.39285714285714285, 0.8928571428571429,
        0.39285714285714285, 0.8928571428571429,
        0.4642857142857143, 0.8928571428571429,
        0.4642857142857143, 0.8928571428571429,
        0.5357142857142857, 0.8928571428571429,
        0.5357142857142857, 0.8928571428571429,
        0.6071428571428571, 0.8928571428571429,
        0.6071428571428571, 0.8928571428571429,
        0.6785714285714286, 0.8928571428571429,
        0.6785714285714286, 0.8928571428571429,
        0.75, 0.8928571428571429,
        0.75, 0.8928571428571429,
        0.8214285714285714, 0.8928571428571429,
        0.8214285714285714, 0.8928571428571429,
        0.8928571428571429, 0.8928571428571429,
        0.8928571428571429, 0.8928571428571429,
        0.9642857142857143, 0.8928571428571429,
        0.9642857142857143, 0.8928571428571429,
        0.03571428571428571, 0.9642857142857143,
        0.03571428571428571, 0.9642857142857143,
        0.10714285714285714, 0.9642857142857143,
        0.10714285714285714, 0.9642857142857143,
        0.17857142857142858, 0.9642857142857143,
        0.17857142857142858, 0.9642857142857143,
        0.25, 0.9642857142857143,
        0.25, 0.9642857142857143,
        0.32142857142857145, 0.9642857142857143,
        0.32142857142857145, 0.9642857142857143,
        0.39285714285714285, 0.9642857142857143,
        0.39285714285714285, 0.9642857142857143,
        0.4642857142857143, 0.9642857142857143,
        0.4642857142857143, 0.9642857142857143,
        0.5357142857142857, 0.9642857142857143,
        0.5357142857142857, 0.9642857142857143,
        0.6071428571428571, 0.9642857142857143,
        0.6071428571428571, 0.9642857142857143,
        0.6785714285714286, 0.9642857142857143,
        0.6785714285714286, 0.9642857142857143,
        0.75, 0.9642857142857143,
        0.75, 0.9642857142857143,
        0.8214285714285714, 0.9642857142857143,
        0.8214285714285714, 0.9642857142857143,
        0.8928571428571429, 0.9642857142857143,
        0.8928571428571429, 0.9642857142857143,
        0.9642857142857143, 0.9642857142857143,
        0.9642857142857143, 0.9642857142857143,
        0.07142857142857142, 0.07142857142857142,
        0.07142857142857142, 0.07142857142857142,
        0.07142857142857142, 0.07142857142857142,
        0.07142857142857142, 0.07142857142857142,
        0.07142857142857142, 0.07142857142857142,
        0.07142857142857142, 0.07142857142857142,
        0.21428571428571427, 0.07142857142857142,
        0.21428571428571427, 0.07142857142857142,
        0.21428571428571427, 0.07142857142857142,
        0.21428571428571427, 0.07142857142857142,
        0.21428571428571427, 0.07142857142857142,
        0.21428571428571427, 0.07142857142857142,
        0.35714285714285715, 0.07142857142857142,
        0.35714285714285715, 0.07142857142857142,
        0.35714285714285715, 0.07142857142857142,
        0.35714285714285715, 0.07142857142857142,
        0.35714285714285715, 0.07142857142857142,
        0.35714285714285715, 0.07142857142857142,
        0.5, 0.07142857142857142,
        0.5, 0.07142857142857142,
        0.5, 0.07142857142857142,
        0.5, 0.07142857142857142,
        0.5, 0.07142857142857142,
        0.5, 0.07142857142857142,
        0.6428571428571429, 0.07142857142857142,
        0.6428571428571429, 0.07142857142857142,
        0.6428571428571429, 0.07142857142857142,
        0.6428571428571429, 0.07142857142857142,
        0.6428571428571429, 0.07142857142857142,
        0.6428571428571429, 0.07142857142857142,
        0.7857142857142857, 0.07142857142857142,
        0.7857142857142857, 0.07142857142857142,
        0.7857142857142857, 0.07142857142857142,
        0.7857142857142857, 0.07142857142857142,
        0.7857142857142857, 0.07142857142857142,
        0.7857142857142857, 0.07142857142857142,
        0.9285714285714286, 0.07142857142857142,
        0.9285714285714286, 0.07142857142857142,
        0.9285714285714286, 0.07142857142857142,
        0.9285714285714286, 0.07142857142857142,
        0.9285714285714286, 0.07142857142857142,
        0.9285714285714286, 0.07142857142857142,
        0.07142857142857142, 0.21428571428571427,
        0.07142857142857142, 0.21428571428571427,
        0.07142857142857142, 0.21428571428571427,
        0.07142857142857142, 0.21428571428571427,
        0.07142857142857142, 0.21428571428571427,
        0.07142857142857142, 0.21428571428571427,
        0.21428571428571427, 0.21428571428571427,
        0.21428571428571427, 0.21428571428571427,
        0.21428571428571427, 0.21428571428571427,
        0.21428571428571427, 0.21428571428571427,
        0.21428571428571427, 0.21428571428571427,
        0.21428571428571427, 0.21428571428571427,
        0.35714285714285715, 0.21428571428571427,
        0.35714285714285715, 0.21428571428571427,
        0.35714285714285715, 0.21428571428571427,
        0.35714285714285715, 0.21428571428571427,
        0.35714285714285715, 0.21428571428571427,
        0.35714285714285715, 0.21428571428571427,
        0.5, 0.21428571428571427,
        0.5, 0.21428571428571427,
        0.5, 0.21428571428571427,
        0.5, 0.21428571428571427,
        0.5, 0.21428571428571427,
        0.5, 0.21428571428571427,
        0.6428571428571429, 0.21428571428571427,
        0.6428571428571429, 0.21428571428571427,
        0.6428571428571429, 0.21428571428571427,
        0.6428571428571429, 0.21428571428571427,
        0.6428571428571429, 0.21428571428571427,
        0.6428571428571429, 0.21428571428571427,
        0.7857142857142857, 0.21428571428571427,
        0.7857142857142857, 0.21428571428571427,
        0.7857142857142857, 0.21428571428571427,
        0.7857142857142857, 0.21428571428571427,
        0.7857142857142857, 0.21428571428571427,
        0.7857142857142857, 0.21428571428571427,
        0.9285714285714286, 0.21428571428571427,
        0.9285714285714286, 0.21428571428571427,
        0.9285714285714286, 0.21428571428571427,
        0.9285714285714286, 0.21428571428571427,
        0.9285714285714286, 0.21428571428571427,
        0.9285714285714286, 0.21428571428571427,
        0.07142857142857142, 0.35714285714285715,
        0.07142857142857142, 0.35714285714285715,
        0.07142857142857142, 0.35714285714285715,
        0.07142857142857142, 0.35714285714285715,
        0.07142857142857142, 0.35714285714285715,
        0.07142857142857142, 0.35714285714285715,
        0.21428571428571427, 0.35714285714285715,
        0.21428571428571427, 0.35714285714285715,
        0.21428571428571427, 0.35714285714285715,
        0.21428571428571427, 0.35714285714285715,
        0.21428571428571427, 0.35714285714285715,
        0.21428571428571427, 0.35714285714285715,
        0.35714285714285715, 0.35714285714285715,
        0.35714285714285715, 0.35714285714285715,
        0.35714285714285715, 0.35714285714285715,
        0.35714285714285715, 0.35714285714285715,
        0.35714285714285715, 0.35714285714285715,
        0.35714285714285715, 0.35714285714285715,
        0.5, 0.35714285714285715,
        0.5, 0.35714285714285715,
        0.5, 0.35714285714285715,
        0.5, 0.35714285714285715,
        0.5, 0.35714285714285715,
        0.5, 0.35714285714285715,
        0.6428571428571429, 0.35714285714285715,
        0.6428571428571429, 0.35714285714285715,
        0.6428571428571429, 0.35714285714285715,
        0.6428571428571429, 0.35714285714285715,
        0.6428571428571429, 0.35714285714285715,
        0.6428571428571429, 0.35714285714285715,
        0.7857142857142857, 0.35714285714285715,
        0.7857142857142857, 0.35714285714285715,
        0.7857142857142857, 0.35714285714285715,
        0.7857142857142857, 0.35714285714285715,
        0.7857142857142857, 0.35714285714285715,
        0.7857142857142857, 0.35714285714285715,
        0.9285714285714286, 0.35714285714285715,
        0.9285714285714286, 0.35714285714285715,
        0.9285714285714286, 0.35714285714285715,
        0.9285714285714286, 0.35714285714285715,
        0.9285714285714286, 0.35714285714285715,
        0.9285714285714286, 0.35714285714285715,
        0.07142857142857142, 0.5,
        0.07142857142857142, 0.5,
        0.07142857142857142, 0.5,
        0.07142857142857142, 0.5,
        0.07142857142857142, 0.5,
        0.07142857142857142, 0.5,
        0.21428571428571427, 0.5,
        0.21428571428571427, 0.5,
        0.21428571428571427, 0.5,
        0.21428571428571427, 0.5,
        0.21428571428571427, 0.5,
        0.21428571428571427, 0.5,
        0.35714285714285715, 0.5,
        0.35714285714285715, 0.5,
        0.35714285714285715, 0.5,
        0.35714285714285715, 0.5,
        0.35714285714285715, 0.5,
        0.35714285714285715, 0.5,
        0.5, 0.5,
        0.5, 0.5,
        0.5, 0.5,
        0.5, 0.5,
        0.5, 0.5,
        0.5, 0.5,
        0.6428571428571429, 0.5,
        0.6428571428571429, 0.5,
        0.6428571428571429, 0.5,
        0.6428571428571429, 0.5,
        0.6428571428571429, 0.5,
        0.6428571428571429, 0.5,
        0.7857142857142857, 0.5,
        0.7857142857142857, 0.5,
        0.7857142857142857, 0.5,
        0.7857142857142857, 0.5,
        0.7857142857142857, 0.5,
        0.7857142857142857, 0.5,
        0.9285714285714286, 0.5,
        0.9285714285714286, 0.5,
        0.9285714285714286, 0.5,
        0.9285714285714286, 0.5,
        0.9285714285714286, 0.5,
        0.9285714285714286, 0.5,
        0.07142857142857142, 0.6428571428571429,
        0.07142857142857142, 0.6428571428571429,
        0.07142857142857142, 0.6428571428571429,
        0.07142857142857142, 0.6428571428571429,
        0.07142857142857142, 0.6428571428571429,
        0.07142857142857142, 0.6428571428571429,
        0.21428571428571427, 0.6428571428571429,
        0.21428571428571427, 0.6428571428571429,
        0.21428571428571427, 0.6428571428571429,
        0.21428571428571427, 0.6428571428571429,
        0.21428571428571427, 0.6428571428571429,
        0.21428571428571427, 0.6428571428571429,
        0.35714285714285715, 0.6428571428571429,
        0.35714285714285715, 0.6428571428571429,
        0.35714285714285715, 0.6428571428571429,
        0.35714285714285715, 0.6428571428571429,
        0.35714285714285715, 0.6428571428571429,
        0.35714285714285715, 0.6428571428571429,
        0.5, 0.6428571428571429,
        0.5, 0.6428571428571429,
        0.5, 0.6428571428571429,
        0.5, 0.6428571428571429,
        0.5, 0.6428571428571429,
        0.5, 0.6428571428571429,
        0.6428571428571429, 0.6428571428571429,
        0.6428571428571429, 0.6428571428571429,
        0.6428571428571429, 0.6428571428571429,
        0.6428571428571429, 0.6428571428571429,
        0.6428571428571429, 0.6428571428571429,
        0.6428571428571429, 0.6428571428571429,
        0.7857142857142857, 0.6428571428571429,
        0.7857142857142857, 0.6428571428571429,
        0.7857142857142857, 0.6428571428571429,
        0.7857142857142857, 0.6428571428571429,
        0.7857142857142857, 0.6428571428571429,
        0.7857142857142857, 0.6428571428571429,
        0.9285714285714286, 0.6428571428571429,
        0.9285714285714286, 0.6428571428571429,
        0.9285714285714286, 0.6428571428571429,
        0.9285714285714286, 0.6428571428571429,
        0.9285714285714286, 0.6428571428571429,
        0.9285714285714286, 0.6428571428571429,
        0.07142857142857142, 0.7857142857142857,
        0.07142857142857142, 0.7857142857142857,
        0.07142857142857142, 0.7857142857142857,
        0.07142857142857142, 0.7857142857142857,
        0.07142857142857142, 0.7857142857142857,
        0.07142857142857142, 0.7857142857142857,
        0.21428571428571427, 0.7857142857142857,
        0.21428571428571427, 0.7857142857142857,
        0.21428571428571427, 0.7857142857142857,
        0.21428571428571427, 0.7857142857142857,
        0.21428571428571427, 0.7857142857142857,
        0.21428571428571427, 0.7857142857142857,
        0.35714285714285715, 0.7857142857142857,
        0.35714285714285715, 0.7857142857142857,
        0.35714285714285715, 0.7857142857142857,
        0.35714285714285715, 0.7857142857142857,
        0.35714285714285715, 0.7857142857142857,
        0.35714285714285715, 0.7857142857142857,
        0.5, 0.7857142857142857,
        0.5, 0.7857142857142857,
        0.5, 0.7857142857142857,
        0.5, 0.7857142857142857,
        0.5, 0.7857142857142857,
        0.5, 0.7857142857142857,
        0.6428571428571429, 0.7857142857142857,
        0.6428571428571429, 0.7857142857142857,
        0.6428571428571429, 0.7857142857142857,
        0.6428571428571429, 0.7857142857142857,
        0.6428571428571429, 0.7857142857142857,
        0.6428571428571429, 0.7857142857142857,
        0.7857142857142857, 0.7857142857142857,
        0.7857142857142857, 0.7857142857142857,
        0.7857142857142857, 0.7857142857142857,
        0.7857142857142857, 0.7857142857142857,
        0.7857142857142857, 0.7857142857142857,
        0.7857142857142857, 0.7857142857142857,
        0.9285714285714286, 0.7857142857142857,
        0.9285714285714286, 0.7857142857142857,
        0.9285714285714286, 0.7857142857142857,
        0.9285714285714286, 0.7857142857142857,
        0.9285714285714286, 0.7857142857142857,
        0.9285714285714286, 0.7857142857142857,
        0.07142857142857142, 0.9285714285714286,
        0.07142857142857142, 0.9285714285714286,
        0.07142857142857142, 0.9285714285714286,
        0.07142857142857142, 0.9285714285714286,
        0.07142857142857142, 0.9285714285714286,
        0.07142857142857142, 0.9285714285714286,
        0.21428571428571427, 0.9285714285714286,
        0.21428571428571427, 0.9285714285714286,
        0.21428571428571427, 0.9285714285714286,
        0.21428571428571427, 0.9285714285714286,
        0.21428571428571427, 0.9285714285714286,
        0.21428571428571427, 0.9285714285714286,
        0.35714285714285715, 0.9285714285714286,
        0.35714285714285715, 0.9285714285714286,
        0.35714285714285715, 0.9285714285714286,
        0.35714285714285715, 0.9285714285714286,
        0.35714285714285715, 0.9285714285714286,
        0.35714285714285715, 0.9285714285714286,
        0.5, 0.9285714285714286,
        0.5, 0.9285714285714286,
        0.5, 0.9285714285714286,
        0.5, 0.9285714285714286,
        0.5, 0.9285714285714286,
        0.5, 0.9285714285714286,
        0.6428571428571429, 0.9285714285714286,
        0.6428571428571429, 0.9285714285714286,
        0.6428571428571429, 0.9285714285714286,
        0.6428571428571429, 0.9285714285714286,
        0.6428571428571429, 0.9285714285714286,
        0.6428571428571429, 0.9285714285714286,
        0.7857142857142857, 0.9285714285714286,
        0.7857142857142857, 0.9285714285714286,
        0.7857142857142857, 0.9285714285714286,
        0.7857142857142857, 0.9285714285714286,
        0.7857142857142857, 0.9285714285714286,
        0.7857142857142857, 0.9285714285714286,
        0.9285714285714286, 0.9285714285714286,
        0.9285714285714286, 0.9285714285714286,
        0.9285714285714286, 0.9285714285714286,
        0.9285714285714286, 0.9285714285714286,
        0.9285714285714286, 0.9285714285714286,
        0.9285714285714286, 0.9285714285714286);
        return anchor;
}
