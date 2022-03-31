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
    print('This version of OpenCV does not support TIM-VX and NPU. Visit https://gist.github.com/fengyuentau/5a7a5ba36328f2b763aea026c43fa45f for more information.')

parser = argparse.ArgumentParser(
    description="An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (https://arxiv.org/abs/1507.05717)")
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='text_recognition_CRNN_EN_2021sep.onnx', help='Path to the model.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--charset', '-c', type=str, default='charset_36_EN.txt', help='Path to the charset file corresponding to the selected model.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
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
    recognizer = CRNN(modelPath=args.model, charsetPath=args.charset)
    # Instantiate DB for text detection
    detector = DB(modelPath='../text_detection_db/text_detection_DB_IC15_resnet18_2021sep.onnx',
                  inputSize=[736, 736],
                  binaryThreshold=0.3,
                  polygonThreshold=0.5,
                  maxCandidates=200,
                  unclipRatio=2.0,
                  backendId=args.backend,
                  targetId=args.target
    )

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)
        image = cv.resize(image, [args.width, args.height])

        # Inference
        results = detector.infer(image)
        texts = []
        for box, score in zip(results[0], results[1]):
            texts.append(
                recognizer.infer(image, box.reshape(8))
            )

        # Draw results on the input image
        image = visualize(image, results, texts)

        # Save results if save is true
        if args.save:
            print('Resutls saved to result.jpg\n')
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

            frame = cv.resize(frame, [736, 736])
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

                # Draw results on the input image
                frame = visualize(frame, results, texts)
                print(texts)

            # Visualize results in a new Window
            cv.imshow('{} Demo'.format(recognizer.name), frame)

