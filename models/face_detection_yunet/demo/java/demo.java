import org.opencv.core.*;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

public class demo {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        String modelPath = "../../face_detection_yunet_2023mar.onnx";
        String imagePath = "../../example_outputs/largest_selfie.jpg";
        FaceDetectorYN detector = FaceDetectorYN.create(modelPath, "", new Size(320, 320), 0.9f, 0.3f, 5000);
        Mat img = Imgcodecs.imread(imagePath);
        if (img.empty()) {
            System.out.println("Could not read the image, opening camera...");
            runCamera(detector);
        } else {
            runImage(detector, img);
        }
    }

    private static void runImage(FaceDetectorYN detector, Mat img) {
        detector.setInputSize(img.size());
        Mat faces = new Mat();
        detector.detect(img, faces);

        visualize(img, faces);
        HighGui.imshow("Yunet Java Demo - Image", img);
        HighGui.waitKey(0);
        System.exit(0);
    }

    private static void runCamera(FaceDetectorYN detector) {
        VideoCapture cap = new VideoCapture(0);
        Mat frame = new Mat();
        if (!cap.isOpened()) {
            System.out.println("Could not open camera");
            return;
        }
        while (true) {
            cap.read(frame);
            if (frame.empty()) {
                System.out.println("No captured frame -- Break!");
                break;
            }
            detector.setInputSize(frame.size());
            Mat faces = new Mat();
            detector.detect(frame, faces);

            visualize(frame, faces);
            HighGui.imshow("Yunet Java Demo - Camera", frame);
            if (HighGui.waitKey(30) >= 0) {
                break;
            }
        }
        cap.release();
    }

    private static void visualize(Mat img, Mat faces) {
        for (int i = 0; i < faces.rows(); i++) {
            float[] data = new float[15];
            faces.get(i, 0, data);

            Imgproc.rectangle(img, new Rect((int) data[0], (int) data[1], (int) data[2], (int) data[3]),
                    new Scalar(0, 255, 0), 2);
            for (int j = 0; j < 5; j++) {
                Point p = new Point(data[4 + j * 2], data[5 + j * 2]);
                Imgproc.circle(img, p, 2, new Scalar(255, 0, 0), 2);
            }
        }
    }
}