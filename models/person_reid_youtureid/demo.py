# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import os
import argparse

import numpy as np
import cv2 as cv

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from youtureid import YoutuReID

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="ReID baseline models from Tencent Youtu Lab")
parser.add_argument('--query_dir', '-q', type=str,
                    help='Query directory.')
parser.add_argument('--gallery_dir', '-g', type=str,
                    help='Gallery directory.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--topk', type=int, default=10,
                    help='Top-K closest from gallery for each query.')
parser.add_argument('--model', '-m', type=str, default='person_reid_youtu_2021nov.onnx',
                    help='Path to the model.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
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
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate YoutuReID for person ReID
    net = YoutuReID(modelPath=args.model, backendId=backend_id, targetId=target_id)

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

