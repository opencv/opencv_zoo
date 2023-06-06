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

# Check OpenCV version
assert cv.__version__ >= "4.7.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition (https://ieeexplore.ieee.org/document/9318547)")
parser.add_argument('--input1', '-i1', type=str,
                    help='Usage: Set path to the input image 1 (original face).')
parser.add_argument('--input2', '-i2', type=str,
                    help='Usage: Set path to the input image 2 (comparison face).')
parser.add_argument('--model', '-m', type=str, default='face_recognition_sface_2021dec.onnx',
                    help='Usage: Set model path, defaults to face_recognition_sface_2021dec.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0,
                    help='Usage: Distance type. \'0\': cosine, \'1\': norm_l1. Defaults to \'0\'')
args = parser.parse_args()

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    # Instantiate SFace for face recognition
    recognizer = SFace(modelPath=args.model,
                       disType=args.dis_type,
                       backendId=backend_id,
                       targetId=target_id)
    # Instantiate YuNet for face detection
    detector = YuNet(modelPath='../face_detection_yunet/face_detection_yunet_2023mar.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.9,
                     nmsThreshold=0.3,
                     topK=5000,
                     backendId=backend_id,
                     targetId=target_id)

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
