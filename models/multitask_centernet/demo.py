import cv2
import argparse
import numpy as np
from multitask_centernet import MCN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/d2645891.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='MCN.onnx')
    args = parser.parse_args()

    mcn = MCN(args.modelpath)
    srcimg = cv2.imread(args.imgpath)
    srcimg = mcn.detect(srcimg)
    cv2.imwrite('result.png', srcimg)
