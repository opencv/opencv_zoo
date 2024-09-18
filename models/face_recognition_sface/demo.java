import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.FaceDetectorYN;
import org.opencv.objdetect.FaceRecognizerSF;

/**
 * the java demo of FaceRecognizerSF
 *
 * you need the dependencies of maven：
 *
 *   *OpenCV Java bindings packaged with native libraries,
 *   seamlessly delivered as a turn-key Maven dependency.
 *   You don't need to download OpenCV to support the relevant features.
 *   doc:https://github.com/openpnp/opencv?tab=readme-ov-file
 *
 *   https://mvnrepository.com/artifact/org.openpnp/opencv
 *   <dependency>
 *       <groupId>org.openpnp</groupId>
 *       <artifactId>opencv</artifactId>
 *       <version>4.9.0-0</version>
 *   </dependency>
 */
public class FaceRecognizer {

    private static double cosine_similar_threshold = 0.363;

    private static double l2norm_similar_threshold = 1.128;

    // Your full path of yunet model
    // <a href="https://docs.opencv.org/4.8.0/df/d20/classcv_1_1FaceDetectorYN.html">FaceDetectorYN</a>
    private static String faceDetectModelPath = "/face_detection_yunet_2023mar.onnx";
    // state faceDetector
    private static FaceDetectorYN faceDetector = null;

    // Your full path of sface model
    private static String faceRecognizModelPath = "/face_recognition_sface_2021dec.onnx";
    // state faceRecognizer
    private static FaceRecognizerSF faceRecognizer = null;

    public static void main(String[] args) {
        // You need to use the full path of img please
        boolean b = faceRecognizer("imgPathA", "imgPathB");
        System.out.println(b);
    }

    public static boolean faceRecognizer(String imgPathA, String imgPathB) {
        // Load for opencv
        OpenCV.loadLocally();
        // Load for faceDetector
        loadFaceDetector();
        // Load for faceRecognizer
        loadFaceRecognizer();

        return faceRecognizerUtil(imgPathA, imgPathB);
    }

    // Load of faceDetector
    private static void loadFaceDetector() {
        if (faceDetector != null) {
            return;
        }
        // You could use the full path for faceDetect model instead to get the resource
        faceDetector = FaceDetectorYN.create(faceDetectModelPath, "", new Size());
    }

    // Load for faceRecognizer
    private static void loadFaceRecognizer() {
        if (faceRecognizer != null) {
            return;
        }
        // You could use the full path for faceRecogniz model instead to get the resource
        faceRecognizer = FaceRecognizerSF.create(faceRecognizModelPath, "");
    }

    /**
     * FaceRecogniz. Calculating the distance between two face features
     *
     * @param imgPathA the path of imgA
     * @param imgPathB the path of imgB
     */
    private static boolean faceRecognizerUtil(String imgPathA, String imgPathB) {
        // 1.Read img convert to a mat
        Mat imgA = Imgcodecs.imread(imgPathA);
        Mat imgB = Imgcodecs.imread(imgPathB);

        // 2.Detect face from given image
        Mat faceA = new Mat();
        faceDetector.setInputSize(imgA.size());
        faceDetector.detect(imgA, faceA);
        Mat faceB = new Mat();
        faceDetector.setInputSize(imgB.size());
        faceDetector.detect(imgB, faceB);

        // 3.Aligning image to put face on the standard position
        Mat alignFaceA = new Mat();
        faceRecognizer.alignCrop(imgA, faceA.row(0), alignFaceA);
        Mat alignFaceB = new Mat();
        faceRecognizer.alignCrop(imgB, faceB.row(0), alignFaceB);

        // 4.Extracting face feature from aligned image
        Mat featureA = new Mat();
        faceRecognizer.feature(alignFaceA, featureA);
        featureA = featureA.clone();
        Mat featureB = new Mat();
        faceRecognizer.feature(alignFaceB, featureB);
        featureB = featureB.clone();

        // 5.FaceRecogniz. Calculating the distance between two face features. If the condition is met, it returns true
        // Get cosine similar
        double match1 = faceRecognizer.match(featureA, featureB, FaceRecognizerSF.FR_COSINE);
        // Get l2norm similar
        double match2 = faceRecognizer.match(featureA, featureB, FaceRecognizerSF.FR_NORM_L2);
        if (match1 >= cosine_similar_threshold && match2 <= l2norm_similar_threshold) {
            return true;
        } else {
            return false;
        }
    }
}
