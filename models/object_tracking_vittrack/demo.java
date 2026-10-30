import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.UnixStyleUsageFormatter;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_video.TrackerVit;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.bytedeco.opencv.opencv_videoio.VideoWriter;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FPS;

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
        @Parameter(names = {"--input", "-i"}, order = 1,
                description = "Set path to the input video. Omit for using default camera.")
        String input;
        @Parameter(names = {"--model_path", "-m"}, order = 2,
                description = "Set model path.")
        String modelPath = "object_tracking_vittrack_2023sep.onnx";
        @Parameter(names = {"--backend_target", "-bt"}, order = 3,
                description = "Choose one of the backend-target pair to run this demo:" +
                        " 0: OpenCV implementation + CPU," +
                        " 1: CUDA + GPU (CUDA), " +
                        " 2: CUDA + GPU (CUDA FP16)," +
                        " 3: TIM-VX + NPU," +
                        " 4: CANN + NPU")
        int backendTarget = 0;
        @Parameter(names = {"--save", "-s"}, order = 4,
                description = "Specify to save a file with results.")
        boolean save;
        @Parameter(names = {"--vis", "-v"}, order = 5, arity = 1,
                description = "Specify to open a new window to show results.")
        boolean vis = true;
    }

    static class TrackingResult {
        boolean isLocated;
        Rect bbox;
        float score;
    }

    static class VitTrack {
        private final TrackerVit model;

        VitTrack(String modelPath, int backendId, int targetId) {
            final TrackerVit.Params params = new TrackerVit.Params();
            params.net(new BytePointer(modelPath))
                    .backend(backendId)
                    .target(targetId);
            model = TrackerVit.create(params);
        }

        void init(Mat image, Rect roi) {
            model.init(image, roi);
        }

        TrackingResult infer(Mat image) {
            final TrackingResult result = new TrackingResult();
            result.bbox = new Rect();
            result.isLocated = model.update(image, result.bbox);
            result.score = model.getTrackingScore();
            return result;
        }
    }

    static Mat visualize(Mat image, Rect bbox, float score, boolean isLocated, double fps, Scalar boxColor,
                         Scalar textColor, double fontScale, int fontSize) {
        final Mat output = image.clone();
        final int h = output.rows();
        final int w = output.cols();
        if (fps >= 0) {
            putText(output, String.format("FPS: %.2f", fps), new Point(0, 30), FONT_HERSHEY_DUPLEX, fontScale,
                    textColor);
        }

        if (isLocated && score >= 0.3) {
            rectangle(output, bbox, boxColor, 2, LINE_8, 0);
            putText(output, String.format("%.2f", score), new Point(bbox.x(), bbox.y() + 25),
                    FONT_HERSHEY_DUPLEX, fontScale, textColor, fontSize, LINE_8, false);
        } else {
            final Size textSize = getTextSize("Target lost!", FONT_HERSHEY_DUPLEX, fontScale, fontSize, new int[]{0});
            final int textX = (w - textSize.width()) / 2;
            final int textY = (h - textSize.height()) / 2;
            putText(output, "Target lost!", new Point(textX, textY), FONT_HERSHEY_DUPLEX,
                    fontScale, new Scalar(0, 0, 255, 0), fontSize, LINE_8, false);
        }

        return output;
    }

    /**
     * Execute: mvn compile exec:java -q -Dexec.args=""
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
        final int backendId = backendTargetPairs[args.backendTarget][0];
        final int targetId = backendTargetPairs[args.backendTarget][1];
        VitTrack tracker = new VitTrack(args.modelPath, backendId, targetId);

        final VideoCapture video = new VideoCapture();
        if (args.input == null) {
            video.open(0);
        } else {
            video.open(args.input);
        }
        if (!video.isOpened()) {
            System.err.println("Error: Could not open video source");
            return;
        }

        Mat firstFrame = new Mat();
        video.read(firstFrame);

        if (firstFrame.empty()) {
            System.err.println("No frames grabbed!");
            return;
        }

        Mat firstFrameCopy = firstFrame.clone();
        putText(firstFrameCopy, "1. Drag a bounding box to track.", new Point(0, 25), FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0, 0));
        putText(firstFrameCopy, "2. Press ENTER to confirm", new Point(0, 50), FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0, 0));
        final Rect roi = selectROI("VitTrack Demo", firstFrameCopy);

        if (roi.area() == 0) {
            System.err.println("No ROI is selected! Exiting...");
            return;
        } else {
            System.out.printf("Selected ROI: (x: %d, y: %d, width: %d, height: %d)%n", roi.x(), roi.y(), roi.width(),
                    roi.height());
        }

        // Create VideoWriter if save option is specified
        final VideoWriter outputVideo = new VideoWriter();
        if (args.save) {
            final Size frameSize = firstFrame.size();
            outputVideo.open("output.mp4", VideoWriter.fourcc((byte) 'm', (byte) 'p', (byte) '4', (byte) 'v'),
                    video.get(CAP_PROP_FPS), frameSize);
            if (!outputVideo.isOpened()) {
                System.err.println("Error: Could not create output video stream");
                return;
            }
        }

        // Initialize tracker with ROI
        tracker.init(firstFrame, roi);

        // Track frame by frame
        final TickMeter tm = new TickMeter();
        while (waitKey(1) < 0) {
            video.read(firstFrame);
            if (firstFrame.empty()) {
                System.out.println("End of video");
                break;
            }

            // Inference
            tm.start();
            final TrackingResult result = tracker.infer(firstFrame);
            tm.stop();

            // Visualize
            Mat frame = firstFrame.clone();
            frame = visualize(frame, result.bbox, result.score, result.isLocated, tm.getFPS(),
                    new Scalar(0, 255, 0, 0), new Scalar(0, 255, 0, 0), 1.0, 1);

            if (args.save) {
                outputVideo.write(frame);
            }
            if (args.vis) {
                imshow("VitTrack Demo", frame);
            }
            tm.reset();
        }
        if (args.save) {
            outputVideo.release();
        }

        video.release();
    }

}
