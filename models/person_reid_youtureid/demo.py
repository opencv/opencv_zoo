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
    description="ReID baseline models from Tencent Youtu Lab")
parser.add_argument('--query_dir', '-q', type=str, help='Query directory.')
parser.add_argument('--gallery_dir', '-g', type=str, help='Gallery directory.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--topk', type=int, default=10, help='Top-K closest from gallery for each query.')
parser.add_argument('--model', '-m', type=str, default='person_reid_youtu_2021nov.onnx', help='Path to the model.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()

def readImageFromDirectory(img_dir, w=128, h=256):
    img_list = []
    file_list = os.listdir(img_dir)
    for f in file_list:
        img = cv.imread(os.path.join(img_dir, f))
        img = cv.resize(img, (w, h))
        img_list.append(img)
    return img_list, file_list

def visualize(results, query_dir, gallery_dir, output_size=(128, 384)):
    def addBorder(img, color, borderSize=5):
        border = cv.copyMakeBorder(img, top=borderSize, bottom=borderSize, left=borderSize, right=borderSize, borderType=cv.BORDER_CONSTANT, value=color)
        return border

    results_vis = dict.fromkeys(results.keys(), None)
    for f, topk_f in results.items():
        query_img = cv.imread(os.path.join(query_dir, f))
        query_img = cv.resize(query_img, output_size)
        query_img = addBorder(query_img, [0, 0, 0])
        cv.putText(query_img, 'Query', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2)

        gallery_img_list = []
        for idx, gallery_f in enumerate(topk_f):
            gallery_img = cv.imread(os.path.join(gallery_dir, gallery_f))
            gallery_img = cv.resize(gallery_img, output_size)
            gallery_img = addBorder(gallery_img, [255, 255, 255])
            cv.putText(gallery_img, 'G{:02d}'.format(idx), (10, 30), cv.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2)
            gallery_img_list.append(gallery_img)

        results_vis[f] = np.concatenate([query_img] + gallery_img_list, axis=1)

    return results_vis

if __name__ == '__main__':
    # Instantiate YoutuReID for person ReID
    net = YoutuReID(modelPath=args.model, backendId=args.backend, targetId=args.target)

    # Read images from dir
    query_img_list, query_file_list = readImageFromDirectory(args.query_dir)
    gallery_img_list, gallery_file_list = readImageFromDirectory(args.gallery_dir)

    # Query
    topk_indices = net.query(query_img_list, gallery_img_list, args.topk)

    # Index to filename
    results = dict.fromkeys(query_file_list, None)
    for f, indices in zip(query_file_list, topk_indices):
        topk_matches = []
        for idx in indices:
            topk_matches.append(gallery_file_list[idx])
        results[f] = topk_matches
        # Print
        print('Query: {}'.format(f))
        print('\tTop-{} from gallery: {}'.format(args.topk, str(topk_matches)))

    # Visualize
    results_vis = visualize(results, args.query_dir, args.gallery_dir)

    if args.save:
        for f, img in results_vis.items():
            cv.imwrite('result-{}'.format(f), img)

    if args.vis:
        for f, img in results_vis.items():
            cv.namedWindow('result-{}'.format(f), cv.WINDOW_AUTOSIZE)
            cv.imshow('result-{}'.format(f), img)
            cv.waitKey(0)
            cv.destroyAllWindows()

