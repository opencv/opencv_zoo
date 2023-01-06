import argparse

import numpy as np
import cv2 as cv

from mobilenet_v1 import MobileNetV1
from mobilenet_v2 import MobileNetV2

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

parser = argparse.ArgumentParser(description='Demo for MobileNet V1 & V2.')
parser.add_argument('--input', '-i', type=str, help='Usage: Set input path to a certain image, omit if using camera.')
parser.add_argument('--model', '-m', type=str, choices=['v1', 'v2', 'v1-q', 'v2-q'], default='v1', help='Usage: Set model type, defaults to image_classification_mobilenetv1_2022apr.onnx (v1).')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--label', '-l', type=str, default='./imagenet_labels.txt', help='Usage: Set path to the different labels that will be used during the detection. Default list found in imagenet_labels.txt')
args = parser.parse_args()

if __name__ == '__main__':
    # Instantiate ResNet
    models = {
        'v1': MobileNetV1(modelPath='./image_classification_mobilenetv1_2022apr.onnx', labelPath=args.label, backendId=args.backend, targetId=args.target),
        'v2': MobileNetV2(modelPath='./image_classification_mobilenetv2_2022apr.onnx', labelPath=args.label, backendId=args.backend, targetId=args.target),
        'v1-q': MobileNetV1(modelPath='./image_classification_mobilenetv1_2022apr-int8-quantized.onnx', labelPath=args.label, backendId=args.backend, targetId=args.target),
        'v2-q': MobileNetV2(modelPath='./image_classification_mobilenetv2_2022apr-int8-quantized.onnx', labelPath=args.label, backendId=args.backend, targetId=args.target)

    }
    model = models[args.model]

    # Read image and get a 224x224 crop from a 256x256 resized
    image = cv.imread(args.input)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, dsize=(256, 256))
    image = image[16:240, 16:240, :]

    # Inference
    result = model.infer(image)

    # Print result
    print('label: {}'.format(result))

