# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import numpy as np
import cv2 as cv

from dasiamrpn import DaSiamRPN

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser(
    description="Distractor-aware Siamese Networks for Visual Object Tracking (https://arxiv.org/abs/1808.06048)")
parser.add_argument('--input', '-i', type=str, help='Path to the input video. Omit for using default camera.')
parser.add_argument('--model_path', type=str, default='object_tracking_dasiamrpn_model_2021nov.onnx', help='Path to dasiamrpn_model.onnx.')
parser.add_argument('--kernel_cls1_path', type=str, default='object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', help='Path to dasiamrpn_kernel_cls1.onnx.')
parser.add_argument('--kernel_r1_path', type=str, default='object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', help='Path to dasiamrpn_kernel_r1.onnx.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def visualize(image, bbox, score, isLocated, fps=None, box_color=(0, 255, 0),text_color=(0, 255, 0), fontScale = 1, fontSize = 1):
    output = image.copy()
    h, w, _ = output.shape

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 30), cv.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)

    if isLocated and score >= 0.6:
        # bbox: Tuple of length 4
        x, y, w, h = bbox
        cv.rectangle(output, (x, y), (x+w, y+h), box_color, 2)
        cv.putText(output, '{:.2f}'.format(score), (x, y+20), cv.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)
    else:
        text_size, baseline = cv.getTextSize('Target lost!', cv.FONT_HERSHEY_DUPLEX, fontScale, fontSize)
        text_x = int((w - text_size[0]) / 2)
        text_y = int((h - text_size[1]) / 2)
        cv.putText(output, 'Target lost!', (text_x, text_y), cv.FONT_HERSHEY_DUPLEX, fontScale, (0, 0, 255), fontSize)

    return output

if __name__ == '__main__':
    # Instantiate DaSiamRPN
    model = DaSiamRPN(
        model_path=args.model_path,
        kernel_cls1_path=args.kernel_cls1_path,
        kernel_r1_path=args.kernel_r1_path
    )

    # Read from args.input
    _input = args.input
    if args.input is None:
        device_id = 0
        _input = device_id
    video = cv.VideoCapture(_input)

    # Select an object
    has_frame, first_frame = video.read()
    if not has_frame:
        print('No frames grabbed!')
        exit()
    first_frame_copy = first_frame.copy()
    cv.putText(first_frame_copy, "1. Drag a bounding box to track.", (0, 15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv.putText(first_frame_copy, "2. Press ENTER to confirm", (0, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    roi = cv.selectROI('DaSiamRPN Demo', first_frame_copy)
    print("Selected ROI: {}".format(roi))

    # Init tracker with ROI
    model.init(first_frame, roi)

    # Track frame by frame
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        has_frame, frame = video.read()
        if not has_frame:
            print('End of video')
            break
        # Inference
        tm.start()
        isLocated, bbox, score = model.infer(frame)
        tm.stop()
        # Visualize
        frame = visualize(frame, bbox, score, isLocated, fps=tm.getFPS())
        cv.imshow('DaSiamRPN Demo', frame)
        tm.reset()