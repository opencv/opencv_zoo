# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import os
import argparse

import numpy as np
import cv2 as cv

from youtureid import YoutuReID

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser(
    description="ReID baseline models from Tencent Youtu Lab")
parser.add_argument('--query_dir', '-q', type=str, help='Query directory.')
parser.add_argument('--gallery_dir', '-g', type=str, help='Gallery directory.')
parser.add_argument('--topk', type=int, default=10, help='Top-K closest from gallery for each query.')
parser.add_argument('--model', '-m', type=str, default='person_reid_youtu_2021nov.onnx', help='Path to the model.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def readImageFromDirectory(img_dir, w=128, h=256):
    img_list = []
    file_list = os.listdir(img_dir)
    for f in file_list:
        img = cv.imread(os.path.join(img_dir, f))
        img = cv.resize(img, (w, h))
        img_list.append(img)
    return img_list

if __name__ == '__main__':
    # Instantiate YoutuReID for person ReID
    net = YoutuReID(modelPath=args.model)

    # Read images from dir
    query_img_list = readImageFromDirectory(args.query_dir)
    gallery_img_list = readImageFromDirectory(args.gallery_dir)

    print(net.query(query_img_list, gallery_img_list, args.topk))