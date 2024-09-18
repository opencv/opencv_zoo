import cv2 as cv
import argparse
from dexined import Dexined

def get_args_parser(func_args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.', default=0, required=False)
    parser.add_argument('--model', help='Path to dexined.onnx', default='edge_detection_dexined_2024sep.onnx', required=False)

    args, _ = parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)

def main(func_args=None):
    args = get_args_parser(func_args)

    dexined = Dexined(modelPath=args.model)

    # Open video or capture from camera
    cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
    if not cap.isOpened():
        print("Failed to open the input video")
        exit(-1)
        
    cv.namedWindow('Input', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('Output', cv.WINDOW_AUTOSIZE)
    cv.moveWindow('Output', 200, 50)

    # Process frames
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, image = cap.read()
        if not hasFrame:
            print("Press any key to exit")
            cv.waitKey(0)
            break
        
        tm.start()
        result = dexined.infer(image)
        tm.stop()
        label = 'Inference time: {:.2f} ms, FPS: {:.2f}'.format(tm.getTimeMilli(), tm.getFPS())

        cv.imshow("Input", image)
        cv.putText(result, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv.imshow("Output", result)

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
