# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import numpy as np
import cv2 as cv

from ppresnet import PPResNet

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

parser = argparse.ArgumentParser(description='Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385, https://github.com/PaddlePaddle/PaddleHub)')
parser.add_argument('--input', '-i', type=str, help='Path to the input image.')
parser.add_argument('--model', '-m', type=str, default='image_classification_ppresnet50_2022jan.onnx', help='Path to the model.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--label', '-l', type=str, default='./imagenet_labels.txt', help='Path to the dataset labels.')
args = parser.parse_args()

if __name__ == '__main__':
    # Instantiate ResNet
    model = PPResNet(modelPath=args.model, labelPath=args.label, backendId=args.backend, targetId=args.target)

    # Read image and get a 224x224 crop from a 256x256 resized
    image = cv.imread(args.input)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, dsize=(256, 256))
    image = image[16:240, 16:240, :]

    # Inference
    result = model.infer(image)

    # Print result
    print('label: {}'.format(result))

