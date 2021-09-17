# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import numpy as np
import cv2 as cv

from db import DB

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser(description='Real-time Scene Text Detection with Differentiable Binarization (https://arxiv.org/abs/1911.08947).')
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='text_detection_db.onnx', help='Path to the model.')
parser.add_argument('--width', type=int, default=736,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=736,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--binary_threshold', type=float, default=0.3, help='Threshold of the binary map.')
parser.add_argument('--polygon_threshold', type=float, default=0.5, help='Threshold of polygons.')
parser.add_argument('--max_candidates', type=int, default=200, help='Max candidates of polygons.')
parser.add_argument('--unclip_ratio', type=np.float64, default=2.0, help=' The unclip ratio of the detected text region, which determines the output size.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), isClosed=True, thickness=2, fps=None):
    output = image.copy()

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    pts = np.array(results[0])
    output = cv.polylines(output, pts, isClosed, box_color, thickness)

    return output

if __name__ == '__main__':
    # Instantiate DB
    model = DB(modelPath=args.model,
               inputSize=[args.width, args.height],
               binaryThreshold=args.binary_threshold,
               polygonThreshold=args.polygon_threshold,
               maxCandidates=args.max_candidates,
               unclipRatio=args.unclip_ratio
    )

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)
        image = cv.resize(image, [args.width, args.height])

        # Inference
        results = model.infer(image)

        # Print results
        print('{} texts detected.'.format(len(results[0])))
        for idx, (bbox, score) in enumerate(zip(results[0], results[1])):
            print('{}: {} {} {} {}, {:.2f}'.format(idx, bbox[0], bbox[1], bbox[2], bbox[3], score[0]))

        # Draw results on the input image
        image = visualize(image, results)

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
            # Inference
            tm.start()
            results = model.infer(frame) # results is a tuple
            tm.stop()

            # Draw results on the input image
            frame = visualize(frame, results, fps=tm.getFPS())

            # Visualize results in a new Window
            cv.imshow('{} Demo'.format(model.name), frame)

            tm.reset()