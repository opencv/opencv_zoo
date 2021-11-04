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

parser = argparse.ArgumentParser(
    description="An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (https://arxiv.org/abs/1507.05717)")
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='text_recognition_CRNN_VGG_BiLSTM_CTC_2021sep.onnx', help='Path to the model.')
parser.add_argument('--width', type=int, default=736,
                    help='The width of input image being sent to the text detector.')
parser.add_argument('--height', type=int, default=736,
                    help='The height of input image being sent to the text detector.')
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
    recognizer = CRNN(modelPath=args.model)
    # Instantiate DB for text detection
    detector = DB(modelPath='../text_detection_db/text_detection_DB_IC15_resnet18_2021sep.onnx',
                  inputSize=[args.width, args.height],
                  binaryThreshold=0.3,
                  polygonThreshold=0.5,
                  maxCandidates=200,
                  unclipRatio=2.0
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

            frame = cv.resize(frame, [args.width, args.height])
            # Inference of text detector
            tm.start()
            results = detector.infer(frame)
            tm.stop()
            latency_detector = tm.getFPS()
            tm.reset()
            # Inference of text recognizer
            texts = []
            tm.start()
            for box, score in zip(results[0], results[1]):
                result = np.hstack(
                    (box.reshape(8), score)
                )
                texts.append(
                    recognizer.infer(frame, result)
                )
            tm.stop()
            latency_recognizer = tm.getFPS()
            tm.reset()

            # Draw results on the input image
            frame = visualize(frame, results, texts)

            cv.putText(frame, 'Latency - {}: {}'.format(detector.name, latency_detector), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv.putText(frame, 'Latency - {}: {}'.format(recognizer.name, latency_recognizer), (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Visualize results in a new Window
            cv.imshow('{} Demo'.format(recognizer.name), frame)