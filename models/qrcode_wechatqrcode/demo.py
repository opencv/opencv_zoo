# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import numpy as np
import cv2 as cv

from wechatqrcode import WeChatQRCode

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser(
    description="WeChat QR code detector for detecting and parsing QR code (https://github.com/opencv/opencv_contrib/tree/master/modules/wechat_qrcode)")
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--detect_prototxt_path', type=str, default='detect_2021sep.prototxt', help='Path to detect.prototxt.')
parser.add_argument('--detect_model_path', type=str, default='detect_2021sep.caffemodel', help='Path to detect.caffemodel.')
parser.add_argument('--sr_prototxt_path', type=str, default='sr_2021sep.prototxt', help='Path to sr.prototxt.')
parser.add_argument('--sr_model_path', type=str, default='sr_2021sep.caffemodel', help='Path to sr.caffemodel.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

if __name__ == '__main__':
    # Instantiate WeChatQRCode
    model = WeChatQRCode(detect_prototxt_path, detect_model_path, sr_prototxt_path, sr_model_path)

    # If input is an image:
    if args.input is not None:
        image = cv.imread(args.input)
        res, points = model.infer(image)
        print(res)
    else:
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
            res, point = model.infer(image)
            tm.stop()
            fps = tm.getFPS()

            print(res)