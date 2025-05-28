# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import numpy as np
import cv2 as cv

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from ppresnet import PPResNet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385, https://github.com/PaddlePaddle/PaddleHub)')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input path to a certain image, omit if using camera.')
parser.add_argument('--model', '-m', type=str, default='image_classification_ppresnet50_2022jan.onnx',
                    help='Usage: Set model path, defaults to image_classification_ppresnet50_2022jan.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--top_k', type=int, default=1,
                    help='Usage: Get top k predictions.')
args = parser.parse_args()

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    top_k = args.top_k
    # Instantiate ResNet
    model = PPResNet(modelPath=args.model, topK=top_k, backendId=backend_id, targetId=target_id)

    # Read image and get a 224x224 crop from a 256x256 resized
    image = cv.imread(args.input)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, dsize=(256, 256))
    image = image[16:240, 16:240, :]

    # Inference
    result = model.infer(image)[0]

    # Print result
    if top_k == 1:
        print(f"Predicted Label: {result[0]}")
    else:
        print("Predicted Top-K Labels (in decreasing confidence):")
        for i, prediction in enumerate(result):
            print(f"({i+1}) {prediction}")
