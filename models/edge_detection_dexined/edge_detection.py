'''
This sample demonstrates edge detection with dexined (DNN based) edge detection techniques.
'''
import os
import cv2 as cv
import argparse
import numpy as np

def get_args_parser(func_args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.', default=0, required=False)
    parser.add_argument('--model', help='Path to dexined.onnnx', default='edge_detection_dexined_2024sep.onnx', required=False)


    args, _ = parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def post_processing(output, shape):
    h, w = shape
    preds = []
    for p in output:
        img = sigmoid(p)
        img = np.squeeze(img)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        img = cv.resize(img, (w, h))
        preds.append(img)
    fuse = preds[-1]
    ave = np.array(preds, dtype=np.float32)
    ave = np.uint8(np.mean(ave, axis=0))
    return fuse, ave

def loadModel(args):
    net = cv.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net

def apply_dexined(model, image):
    out = model.forward()
    result,_ = post_processing(out, image.shape[:2])
    t, _ = model.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv.putText(result, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv.imshow("Output", result)

def main(func_args=None):
    args = get_args_parser(func_args)

    cap = cv.VideoCapture(cv.samples.findFile(args.input) if args.input else 0)
    if not cap.isOpened():
        print("Failed to open the input video")
        exit(-1)
    cv.namedWindow('Input', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('Output', cv.WINDOW_AUTOSIZE)
    cv.moveWindow('Output', 200, 50)

    net = loadModel(args)
    while cv.waitKey(1) < 0:
        hasFrame, image = cap.read()
        if not hasFrame:
            print("Press any key to exit")
            cv.waitKey(0)
            break
        inp = cv.dnn.blobFromImage(image, 1.0, (512, 512), (103.5, 116.2, 123.6), swapRB=False, crop=False)

        net.setInput(inp)
        apply_dexined(net, image)

        cv.imshow("Input", image)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()