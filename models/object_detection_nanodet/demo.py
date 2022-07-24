import cv2
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt


from Nanodet_m_plus import nanodet

backends = [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_BACKEND_CUDA]
targets = [cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16]
help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
try:
    backends += [cv2.dnn.DNN_BACKEND_TIMVX]
    targets += [cv2.dnn.DNN_TARGET_NPU]
    help_msg_backends += "; {:d}: TIMVX"
    help_msg_targets += "; {:d}: NPU"
except:
    print('This version of OpenCV does not support TIM-VX and NPU. Visit https://gist.github.com/Sidd1609/5bb321c8733110ed613ec120c7c02e41 for more information.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--image_path', type=str, default='test2.jpg', help="image path")
    parser.add_argument('--confidence', default=0.35, type=float, help='class confidence')
    parser.add_argument('--nms', default=0.6, type=float, help='enter nms IOU threshold')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    image = cv2. cvtColor(image, cv2.COLOR_BGR2RGB)
    model_net = nanodet(prob_threshold=args.confidence, iou_threshold=args.nms)

    a = time.time()
    image = model_net.detect(image)
    b = time.time()
    print('Inference_Time:'+str(b-a)+' secs')

    plt.imshow(image)
    plt.show()
