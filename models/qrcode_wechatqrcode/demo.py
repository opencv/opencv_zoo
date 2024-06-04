# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import numpy as np
import cv2 as cv

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from wechatqrcode import WeChatQRCode

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="WeChat QR code detector for detecting and parsing QR code (https://github.com/opencv/opencv_contrib/tree/master/modules/wechat_qrcode)")
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using default camera.')
parser.add_argument('--detect_prototxt_path', type=str, default='detect_2021nov.prototxt',
                    help='Usage: Set path to detect.prototxt.')
parser.add_argument('--detect_model_path', type=str, default='detect_2021nov.caffemodel',
                    help='Usage: Set path to detect.caffemodel.')
parser.add_argument('--sr_prototxt_path', type=str, default='sr_2021nov.prototxt',
                    help='Usage: Set path to sr.prototxt.')
parser.add_argument('--sr_model_path', type=str, default='sr_2021nov.caffemodel',
                    help='Usage: Set path to sr.caffemodel.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

def visualize(image, res, points, points_color=(0, 255, 0), text_color=(0, 255, 0), fps=None):
    output = image.copy()
    h, w, _ = output.shape

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    fontScale = 0.5
    fontSize = 1
    for r, p in zip(res, points):
        p = p.astype(np.int32)
        for _p in p:
            cv.circle(output, _p, 10, points_color, -1)

        qrcode_center_x = int((p[0][0] + p[2][0]) / 2)
        qrcode_center_y = int((p[0][1] + p[2][1]) / 2)

        text_size, baseline = cv.getTextSize(r, cv.FONT_HERSHEY_DUPLEX, fontScale, fontSize)
        text_x = qrcode_center_x - int(text_size[0] / 2)
        text_y = qrcode_center_y - int(text_size[1] / 2)
        cv.putText(output, '{}'.format(r), (text_x, text_y), cv.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)

    return output


if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate WeChatQRCode
    model = WeChatQRCode(args.detect_prototxt_path,
        args.detect_model_path,
        args.sr_prototxt_path,
        args.sr_model_path,
        backendId=backend_id,
        targetId=target_id)

    # If input is an image:
    if args.input is not None:
        image = cv.imread(args.input)
        res, points = model.infer(image)

        # Print results:
        print(res)
        print(points)

        # Draw results on the input image
        image = visualize(image, res, points)

        # Save results if save is true
        if args.save:
            print('Results saved to result.jpg\n')
            cv.imwrite('result.jpg', image)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, image)
            cv.waitKey(0)
    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Inference
            tm.start()
            res, points = model.infer(frame)
            tm.stop()
            fps = tm.getFPS()

            # Draw results on the input image
            frame = visualize(frame, res, points, fps=fps)

            # Visualize results in a new window
            cv.imshow('WeChatQRCode Demo', frame)

            tm.reset()
