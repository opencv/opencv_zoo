import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.UnixStyleUsageFormatter;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.TextDetectionModel_DB;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;

import java.util.AbstractMap;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class demo {

    // Valid combinations of backends and targets
    static int[][] backendTargetPairs = {
            {opencv_dnn.DNN_BACKEND_OPENCV, opencv_dnn.DNN_TARGET_CPU},
            {opencv_dnn.DNN_BACKEND_CUDA, opencv_dnn.DNN_TARGET_CUDA},
            {opencv_dnn.DNN_BACKEND_CUDA, opencv_dnn.DNN_TARGET_CUDA_FP16},
            {opencv_dnn.DNN_BACKEND_TIMVX, opencv_dnn.DNN_TARGET_NPU},
            {opencv_dnn.DNN_BACKEND_CANN, opencv_dnn.DNN_TARGET_NPU}
    };

    static class Args {
        @Parameter(names = {"--help", "-h"}, order = 0, help = true,
                description = "Print help message.")
        boolean help;
        @Parameter(names = {"--model", "-m"}, order = 1,
                description = "Set model type.")
        String model = "text_detection_en_ppocrv3_2023may.onnx";
        @Parameter(names = {"--input", "-i"}, order = 2,
                description = "Path to input image or video file. Skip this argument to capture frames from a camera.")
        String input;
        @Parameter(names = "--width", order = 3,
                description = "Resize input image to certain width, It should be multiple by 32.")
        int width = 736;
        @Parameter(names = "--height", order = 4,
                description = "Resize input image to certain height, It should be multiple by 32.")
        int height = 736;
        @Parameter(names = "--binary_threshold", order = 5,
                description = "Threshold of the binary map.")
        float binaryThreshold = 0.3f;
        @Parameter(names = "--polygon_threshold", order = 6,
                description = "Threshold of polygons.")
        float polygonThreshold = 0.5f;
        @Parameter(names = "--max_candidates", order = 7,
                description = "Set maximum number of polygon candidates.")
        int maxCandidates = 200;
        @Parameter(names = "--unclip_ratio", order = 8,
                description = "The unclip ratio of the detected text region, which determines the output size.")
        double unclipRatio = 2.0;
        @Parameter(names = {"--save", "-s"}, order = 9, arity = 1,
                description = "Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.")
        boolean save = true;
        @Parameter(names = {"--viz", "-v"}, order = 10, arity = 1,
                description = "Specify to open a new window to show results. Invalid in case of camera input.")
        boolean viz = true;
        @Parameter(names = {"--backend", "-bt"}, order = 11,
                description = "Choose one of computation backends:" +
                        " 0: OpenCV implementation + CPU," +
                        " 1: CUDA + GPU (CUDA), " +
                        " 2: CUDA + GPU (CUDA FP16)," +
                        " 3: TIM-VX + NPU," +
                        " 4: CANN + NPU")
        int backend = 0;
    }

    static class PPOCRDet {
        private final TextDetectionModel_DB model;
        private final Size inputSize;

        public PPOCRDet(String modelPath, Size inputSize,
                        float binaryThreshold, float polygonThreshold, int maxCandidates, double unclipRatio,
                        int backendId, int targetId) {
            this.inputSize = inputSize;

            model = new TextDetectionModel_DB(modelPath);
            model.setPreferableBackend(backendId);
            model.setPreferableTarget(targetId);

            model.setBinaryThreshold(binaryThreshold);
            model.setPolygonThreshold(polygonThreshold);
            model.setUnclipRatio(unclipRatio);
            model.setMaxCandidates(maxCandidates);

            model.setInputParams(1.0 / 255.0, inputSize,
                    new Scalar(122.67891434, 116.66876762, 104.00698793, 0), true, false);
        }

        public Map.Entry<PointVectorVector, FloatPointer> infer(Mat image) {
            assert image.rows() == inputSize.height() : "height of input image != net input size";
            assert image.cols() == inputSize.width() : "width of input image != net input size";
            final PointVectorVector pt = new PointVectorVector();
            final FloatPointer confidences = new FloatPointer();
            model.detect(image, pt, confidences);
            return new AbstractMap.SimpleEntry<>(pt, confidences);
        }
    }

    static Mat visualize(Mat image, Map.Entry<PointVectorVector, FloatPointer> results, double fps, Scalar boxColor,
                         Scalar textColor, boolean isClosed, int thickness) {
        final Mat output = new Mat();
        image.copyTo(output);
        if (fps > 0) {
            putText(output, String.format("FPS: %.2f", fps), new Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, textColor);
        }
        final PointVectorVector pvv = results.getKey();
        final MatVector matVector = new MatVector();
        for (int i = 0; i < pvv.size(); i++) {
            final PointVector pv = pvv.get(i);
            final Point pts = new Point(pv.size());
            for (int j = 0; j < pv.size(); j++) {
                pts.position(j).x(pv.get(j).x()).y(pv.get(j).y());
            }
            matVector.push_back(new Mat(pts.position(0)));
        }
        polylines(output, matVector, isClosed, boxColor, thickness, LINE_AA, 0);
        matVector.close();
        return output;
    }

    /**
     * Execute:
     * mvn compile exec:java -Dexec.mainClass=demo -q -Dexec.args="--help"
     */
    public static void main(String[] argv) {
        final Args args = new Args();
        final JCommander jc = JCommander.newBuilder()
                .addObject(args)
                .build();
        jc.setUsageFormatter(new UnixStyleUsageFormatter(jc));
        jc.parse(argv);
        if (args.help) {
            jc.usage();
            return;
        }
        final int[] backendTargetPair = backendTargetPairs[args.backend];
        assert args.model != null && !args.model.isEmpty() : "Model name is empty";
        final Size inpSize = new Size(args.width, args.height);

        final PPOCRDet model = new PPOCRDet(args.model, inpSize,
                args.binaryThreshold, args.polygonThreshold, args.maxCandidates, args.unclipRatio,
                backendTargetPair[0], backendTargetPair[1]);

        final VideoCapture cap = new VideoCapture();
        if (args.input != null) {
            cap.open(args.input);
        } else {
            cap.open(0);
        }
        assert cap.isOpened() : "Cannot open video or file";
        Mat originalImage = new Mat();

        final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        CanvasFrame mainframe = null;
        if (args.input == null || args.viz) {
            mainframe = new CanvasFrame(args.model + " Demo", CanvasFrame.getDefaultGamma() / 2.2);
            mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
            mainframe.setVisible(true);
        }

        final Scalar boxColor = new Scalar(0, 255, 0, 0);
        final Scalar textColor = new Scalar(0, 0, 255, 0);
        final TickMeter tm = new TickMeter();
        while (cap.read(originalImage)) {
            cap.read(originalImage);

            final int originalW = originalImage.cols();
            final int originalH = originalImage.rows();
            final double scaleHeight = originalH / (double) inpSize.height();
            final double scaleWidth = originalW / (double) inpSize.width();
            final Mat image = new Mat();
            resize(originalImage, image, inpSize);

            // inference
            tm.start();
            Map.Entry<PointVectorVector, FloatPointer> results = model.infer(image);
            tm.stop();
            // Scale the results bounding box
            final PointVectorVector pvv = results.getKey();
            for (int i = 0; i < pvv.size(); i++) {
                final PointVector pts = pvv.get(i);
                for (int j = 0; j < pts.size(); j++) {
                    pts.get(j).x((int) (pts.get(j).x() * scaleWidth));
                    pts.get(j).y((int) (pts.get(j).y() * scaleHeight));
                }
            }

            originalImage = visualize(originalImage, results, tm.getFPS(), boxColor, textColor, true, 2);
            tm.reset();
            if (args.input != null) {
                if (args.save) {
                    System.out.println("Result image saved to result.jpg");
                    imwrite("result.jpg", originalImage);
                }
                if (args.viz) {
                    mainframe.showImage(converter.convert(originalImage));
                }
            } else {
                mainframe.showImage(converter.convert(originalImage));
            }

            // clear
            pvv.close();
            image.close();
        }
        tm.close();
    }

}
