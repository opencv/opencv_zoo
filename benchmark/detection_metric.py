# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#

import argparse
import glob
import time

import numpy as np
import cv2 as cv
from cal_rescall.script import cal_recall_precison_f1




def detect_metric(model,**kwargs):
    backend_id = kwargs.pop('backend', 'default')
    data=kwargs['data']
    img_dir=data.pop('imgs')
    gt_dir=data.pop('gt')
    out_dir=data.pop('out')
    width=data.pop('width')
    height=data.pop('height')
    iou_constraint=data.pop("iou_constraint")
    area_precision_constraint=data.pop("area_precision_constraint")
    available_backends = dict(
        default=cv.dnn.DNN_BACKEND_DEFAULT,
        # halide=cv.dnn.DNN_BACKEND_HALIDE,
        # inference_engine=cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
        opencv=cv.dnn.DNN_BACKEND_OPENCV,
        # vkcom=cv.dnn.DNN_BACKEND_VKCOM,
        cuda=cv.dnn.DNN_BACKEND_CUDA,
    )

    target_id = kwargs.pop('target', 'cpu')
    available_targets = dict(
        cpu=cv.dnn.DNN_TARGET_CPU,
        cuda=cv.dnn.DNN_TARGET_CUDA,
        cuda_fp16=cv.dnn.DNN_TARGET_CUDA_FP16,
    )

    # add extra backends & targets
    try:
        available_backends['timvx'] = cv.dnn.DNN_BACKEND_TIMVX
        available_targets['npu'] = cv.dnn.DNN_TARGET_NPU
    except:
        print(
            'OpenCV is not compiled with TIM-VX backend enbaled. See https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU for more details on how to enable TIM-VX backend.')

    _backend = available_backends[backend_id]
    _target = available_targets[target_id]
    model.setBackend(_backend)
    model.setTarget(_target)
    default_evaluation_params = {
        'IOU_CONSTRAINT': iou_constraint,
        'AREA_PRECISION_CONSTRAINT': area_precision_constraint,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        'CRLF': False,  # Lines are delimited by Windows CRLF format
        'CONFIDENCES': True,  # Detections must include confidence value. AP will be calculated
        'PER_SAMPLE_RESULTS': False  # Generate per sample results and produce data for visualization
    }
    # Instantiate DB

    files = glob.glob(img_dir + '/*', recursive=True)

    for file in files:
        image = cv.imread(file)
        scale = (image.shape[1] * 1.0 / width, image.shape[0] * 1.0 / height)
        image = cv.resize(image, [width, height])
        # Inference
        results = model.infer(image)
        img_name = file.split('/')[-1].split('.')[0]
        text_file = out_dir + '/res_' + img_name + '.txt'
        result = ''
        for idx, (bbox, score) in enumerate(zip(results[0], results[1])):
            result += '{},{},{},{},{},{},{},{},{}\n'.format(int(bbox[0][0] * scale[0]), int(bbox[0][1] * scale[1]),
                                                            int(bbox[1][0] * scale[0]), int(bbox[1][1] * scale[1]),
                                                            int(bbox[2][0] * scale[0]), int(bbox[2][1] * scale[1]),
                                                            int(bbox[3][0] * scale[0]), int(bbox[3][1] * scale[1]),
                                                            score)
        with open(text_file, 'w+') as fid:
            fid.write(result)
    result_dict = cal_recall_precison_f1(gt_dir, out_dir, default_evaluation_params)
    print(result_dict)
    return result_dict
