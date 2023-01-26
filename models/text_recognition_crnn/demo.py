# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import sys
import argparse

import numpy as np
import cv2 as cv

from crnn import CRNN

sys.path.append('../text_detection_db')
from db import DB

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

backends = [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_CUDA]
targets = [cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16]
help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
try:
    backends += [cv.dnn.DNN_BACKEND_TIMVX]
    targets += [cv.dnn.DNN_TARGET_NPU]
    help_msg_backends += "; {:d}: TIMVX"
    help_msg_targets += "; {:d}: NPU"
except:
    print('This version of OpenCV does not support TIM-VX and NPU. Visit https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU for more information.')

parser = argparse.ArgumentParser(
    description="An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (https://arxiv.org/abs/1507.05717)")
parser.add_argument('--input', '-i', type=str, help='Usage: Set path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='text_recognition_CRNN_EN_2021sep.onnx', help='Usage: Set model path, defaults to text_recognition_CRNN_EN_2021sep.onnx.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--save', '-s', type=str, default=False, help='Usage: Set “True” to save a file with results. Invalid in case of camera input. Default will be set to “False”.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Usage: Default will be set to “True” and will open a new window to show results. Set to “False” to stop visualizations from being shown. Invalid in case of camera input.')
parser.add_argument('--width', type=int, default=736,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=736,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
args = parser.parse_args()

def visualize(image, boxes, texts, color=(0, 255, 0), isClosed=True, thickness=2):
    output = image.copy()

    pts = np.array(boxes[0])
    output = cv.polylines(output, pts, isClosed, color, thickness)
    for box, text in zip(boxes[0], texts):
        cv.putText(output, text, (box[1].astype(np.int32)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    return output

if __name__ == '__main__':
    # Instantiate CRNN for text recognition
    recognizer = CRNN(modelPath=args.model)
    # Instantiate DB for text detection
    detector = DB(modelPath='../text_detection_db/text_detection_DB_IC15_resnet18_2021sep.onnx',
                  inputSize=[args.width, args.height],
                  binaryThreshold=0.3,
                  polygonThreshold=0.5,
                  maxCandidates=200,
                  unclipRatio=2.0,
                  backendId=args.backend,
                  targetId=args.target
    )

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
            print('Resutls saved to result.jpg\n')
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

