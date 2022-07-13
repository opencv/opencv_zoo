# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse
import glob
import sys
import time

import numpy as np
import cv2 as cv

from db import DB
from cal_rescall.script import cal_recall_precison_f1


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

parser = argparse.ArgumentParser(description='Real-time Scene Text Detection with Differentiable Binarization (https://arxiv.org/abs/1911.08947).')
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='text_detection_DB_TD500_resnet18_2021sep.onnx', help='Path to the model.')
parser.add_argument('--gt_dir', type=str, default='icdar2015/test_gts', help='Path to the ground truth txt directory.')
parser.add_argument('--out_dir', type=str, default='icdar2015/test_predicts', help='Path to the output txt directory.')
parser.add_argument('--img_dir', type=str, default='icdar2015/test_images', help='Path to the test images directory.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--width', type=int, default=736,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=736,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--binary_threshold', type=float, default=0.3, help='Threshold of the binary map.')
parser.add_argument('--polygon_threshold', type=float, default=0.5, help='Threshold of polygons.')
parser.add_argument('--iou_constraint', type=float, default=0.5, help='IOU constraint.')
parser.add_argument('--area_precision_constraint', type=float, default=0.3, help='Area precision constraint.')
parser.add_argument('--max_candidates', type=int, default=200, help='Max candidates of polygons.')
parser.add_argument('--unclip_ratio', type=np.float64, default=2.0, help=' The unclip ratio of the detected text region, which determines the output size.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()



if __name__ == '__main__':
    default_evaluation_params={
        'IOU_CONSTRAINT': args.iou_constraint,
        'AREA_PRECISION_CONSTRAINT': args.area_precision_constraint,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        'CRLF': False,  # Lines are delimited by Windows CRLF format
        'CONFIDENCES': True,  # Detections must include confidence value. AP will be calculated
        'PER_SAMPLE_RESULTS':False  # Generate per sample results and produce data for visualization
    }
    # Instantiate DB
    model = DB(modelPath=args.model,
               inputSize=[args.width, args.height],
               binaryThreshold=args.binary_threshold,
               polygonThreshold=args.polygon_threshold,
               maxCandidates=args.max_candidates,
               unclipRatio=args.unclip_ratio,
               backendId=args.backend,
               targetId=args.target
    )
    files = glob.glob(args.img_dir+'/*', recursive=True)
    start = time.time()
    for file in files:
        image = cv.imread(file)
        image = cv.resize(image, [args.width, args.height])

        # Inference
        results = model.infer(image)
        img_name=file.split('/')[-1].split('.')[0]
        text_file = args.out_dir+'res_' + img_name + '.txt'
        result=''
        for idx, (bbox, score) in enumerate(zip(results[0], results[1])):
            result+='{},{},{},{},{},{},{},{},{}\n'.format(bbox[0][0],bbox[0][1], bbox[1][0], bbox[1][1], bbox[2][0],bbox[2][1], bbox[3][0], bbox[3][1],score)
        with open(text_file, 'w+') as fid:
            fid.write(result)
    end = time.time()
    avg_time=(end-start)/len(files)
    result_dict = cal_recall_precison_f1(args.gt_dir,args.out_dir,default_evaluation_params)
    # Print results
    result_dict['avg time']=avg_time
    print(result_dict)


