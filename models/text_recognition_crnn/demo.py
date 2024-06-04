# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import sys
import argparse

import numpy as np
import cv2 as cv

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from crnn import CRNN

sys.path.append('../text_detection_ppocr')
from ppocr_det import PPOCRDet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (https://arxiv.org/abs/1507.05717)")
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='text_recognition_CRNN_EN_2021sep.onnx',
                    help='Usage: Set model path, defaults to text_recognition_CRNN_EN_2021sep.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--width', type=int, default=736,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=736,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save a file with results. Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

def visualize(image, boxes, texts, color=(0, 255, 0), isClosed=True, thickness=2):
    output = image.copy()

    pts = np.array(boxes[0])
    output = cv.polylines(output, pts, isClosed, color, thickness)
    for box, text in zip(boxes[0], texts):
        cv.putText(output, text, (box[1].astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    return output

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate PPOCRDet for text detection
    detector = PPOCRDet(modelPath='../text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx',
                  inputSize=[args.width, args.height],
                  binaryThreshold=0.3,
                  polygonThreshold=0.5,
                  maxCandidates=200,
                  unclipRatio=2.0,
                  backendId=backend_id,
                  targetId=target_id)
    # Instantiate CRNN for text recognition
    recognizer = CRNN(modelPath=args.model, backendId=backend_id, targetId=target_id)

    # If input is an image
    if args.input is not None:
        original_image = cv.imread(args.input)
        original_w = original_image.shape[1]
        original_h = original_image.shape[0]
        scaleHeight = original_h / args.height
        scaleWidth = original_w / args.width
        image = cv.resize(original_image, [args.width, args.height])

        # Inference
        results = detector.infer(image)
        texts = []
        for box, score in zip(results[0], results[1]):
            texts.append(
                recognizer.infer(image, box.reshape(8))
            )

        # Scale the results bounding box
        for i in range(len(results[0])):
            for j in range(4):
                box = results[0][i][j]
                results[0][i][j][0] = box[0] * scaleWidth
                results[0][i][j][1] = box[1] * scaleHeight

        # Draw results on the input image
        original_image = visualize(original_image, results, texts)

        # Save results if save is true
        if args.save:
            print('Results saved to result.jpg\n')
            cv.imwrite('result.jpg', original_image)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, original_image)
            cv.waitKey(0)
    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, original_image = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            original_w = original_image.shape[1]
            original_h = original_image.shape[0]
            scaleHeight = original_h / args.height
            scaleWidth = original_w / args.width

            frame = cv.resize(original_image, [args.width, args.height])
            # Inference of text detector
            tm.start()
            results = detector.infer(frame)
            tm.stop()
            cv.putText(frame, 'Latency - {}: {:.2f}'.format(detector.name, tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            tm.reset()

            # Inference of text recognizer
            if len(results[0]) and len(results[1]):
                texts = []
                tm.start()
                for box, score in zip(results[0], results[1]):
                    result = np.hstack(
                        (box.reshape(8), score)
                    )
                    texts.append(
                        recognizer.infer(frame, box.reshape(8))
                    )
                tm.stop()
                cv.putText(frame, 'Latency - {}: {:.2f}'.format(recognizer.name, tm.getFPS()), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                tm.reset()

                # Scale the results bounding box
                for i in range(len(results[0])):
                    for j in range(4):
                        box = results[0][i][j]
                        results[0][i][j][0] = box[0] * scaleWidth
                        results[0][i][j][1] = box[1] * scaleHeight

                # Draw results on the input image
                original_image = visualize(original_image, results, texts)
                print(texts)

            # Visualize results in a new Window
            cv.imshow('{} Demo'.format(recognizer.name), original_image)
