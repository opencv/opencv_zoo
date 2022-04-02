# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import sys
import argparse

import numpy as np
import cv2 as cv

from sface import SFace

sys.path.append('../face_detection_yunet')
from yunet import YuNet

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
    description="SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition (https://ieeexplore.ieee.org/document/9318547)")
parser.add_argument('--input1', '-i1', type=str, help='Path to the input image 1.')
parser.add_argument('--input2', '-i2', type=str, help='Path to the input image 2.')
parser.add_argument('--model', '-m', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the model.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0, help='Distance type. \'0\': cosine, \'1\': norm_l1.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

if __name__ == '__main__':
    # Instantiate SFace for face recognition
    recognizer = SFace(modelPath=args.model, disType=args.dis_type, backendId=args.backend, targetId=args.target)
    # Instantiate YuNet for face detection
    detector = YuNet(modelPath='../face_detection_yunet/face_detection_yunet_2022mar.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.9,
                     nmsThreshold=0.3,
                     topK=5000,
                     backendId=args.backend,
                     targetId=args.target)

    img1 = cv.imread(args.input1)
    img2 = cv.imread(args.input2)

    # Detect faces
    detector.setInputSize([img1.shape[1], img1.shape[0]])
    face1 = detector.infer(img1)
    assert face1.shape[0] > 0, 'Cannot find a face in {}'.format(args.input1)
    detector.setInputSize([img2.shape[1], img2.shape[0]])
    face2 = detector.infer(img2)
    assert face2.shape[0] > 0, 'Cannot find a face in {}'.format(args.input2)

    # Match
    result = recognizer.match(img1, face1[0][:-1], img2, face2[0][:-1])
    print('Result: {}.'.format('same identity' if result else 'different identities'))

