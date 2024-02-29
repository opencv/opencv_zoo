# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import os
import sys
import numpy as np
import cv2 as cv

import onnx
from onnx import version_converter
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, quant_pre_process

from transform import Compose, Resize, CenterCrop, Normalize, ColorConvert, HandAlign

class DataReader(CalibrationDataReader):
    def __init__(self, model_path, image_dir, transforms, data_dim):
        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name
        self.transforms = transforms
        self.data_dim = data_dim
        self.data = self.get_calibration_data(image_dir)
        self.enum_data_dicts = iter([{self.input_name: x} for x in self.data])

    def get_next(self):
        return next(self.enum_data_dicts, None)

    def get_calibration_data(self, image_dir):
        blobs = []
        supported = ["jpg", "png"]  # supported file suffix
        for image_name in os.listdir(image_dir):
            image_name_suffix = image_name.split('.')[-1].lower()
            if image_name_suffix not in supported:
                continue
            img = cv.imread(os.path.join(image_dir, image_name))
            img = self.transforms(img)
            if img is None:
                continue
            blob = cv.dnn.blobFromImage(img)
            if self.data_dim == 'hwc':
                blob = cv.transposeND(blob, [0, 2, 3, 1])
            blobs.append(blob)
        return blobs

class Quantize:
    def __init__(self, model_path, calibration_image_dir, transforms=Compose(), per_channel=False, act_type='int8', wt_type='int8', data_dim='chw', nodes_to_exclude=[]):
        self.type_dict = {"uint8" : QuantType.QUInt8, "int8" : QuantType.QInt8}

        self.model_path = model_path
        self.calibration_image_dir = calibration_image_dir
        self.transforms = transforms
        self.per_channel = per_channel
        self.act_type = act_type
        self.wt_type = wt_type
        self.nodes_to_exclude = nodes_to_exclude

        # data reader
        self.dr = DataReader(self.model_path, self.calibration_image_dir, self.transforms, data_dim)

    def check_opset(self):
        model = onnx.load(self.model_path)
        if model.opset_import[0].version != 13:
            print('\tmodel opset version: {}. Converting to opset 13'.format(model.opset_import[0].version))
            # convert opset version to 13
            model_opset13 = version_converter.convert_version(model, 13)
            # save converted model
            output_name = '{}-opset13.onnx'.format(self.model_path[:-5])
            onnx.save_model(model_opset13, output_name)
            # update model_path for quantization
            return output_name
        return self.model_path

    def run(self):
        print('Quantizing {}: act_type {}, wt_type {}'.format(self.model_path, self.act_type, self.wt_type))
        new_model_path = self.check_opset()
        quant_pre_process(new_model_path, new_model_path)
        output_name = '{}_{}.onnx'.format(self.model_path[:-5], self.wt_type)
        quantize_static(new_model_path, output_name, self.dr,
                        quant_format=QuantFormat.QOperator, # start from onnxruntime==1.11.0, quant_format is set to QuantFormat.QDQ by default, which performs fake quantization
                        per_channel=self.per_channel,
                        weight_type=self.type_dict[self.wt_type],
                        activation_type=self.type_dict[self.act_type],
                        nodes_to_exclude=self.nodes_to_exclude)
        if new_model_path != self.model_path:
            os.remove(new_model_path)
        print('\tQuantized model saved to {}'.format(output_name))

models=dict(
    yunet=Quantize(model_path='../../models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
                   calibration_image_dir='../../benchmark/data/face_detection',
                   transforms=Compose([Resize(size=(160, 120))]),
                   nodes_to_exclude=['MaxPool_5', 'MaxPool_18', 'MaxPool_25', 'MaxPool_32'],
    ),
    sface=Quantize(model_path='../../models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
                   calibration_image_dir='../../benchmark/data/face_recognition',
                   transforms=Compose([Resize(size=(112, 112))])),
    pphumanseg=Quantize(model_path='../../models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx',
                        calibration_image_dir='../../benchmark/data/human_segmentation',
                        transforms=Compose([Resize(size=(192, 192))])),
    ppresnet50=Quantize(model_path='../../models/image_classification_ppresnet/image_classification_ppresnet50_2022jan.onnx',
                        calibration_image_dir='../../benchmark/data/image_classification',
                        transforms=Compose([Resize(size=(224, 224))])),
    # TBD: VitTrack
    youtureid=Quantize(model_path='../../models/person_reid_youtureid/person_reid_youtu_2021nov.onnx',
                       calibration_image_dir='../../benchmark/data/person_reid',
                       transforms=Compose([Resize(size=(128, 256))])),
    ppocrv3det_en=Quantize(model_path='../../models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx',
                          calibration_image_dir='../../benchmark/data/text',
                          transforms=Compose([Resize(size=(736, 736)),
                                              Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])])),
    ppocrv3det_cn=Quantize(model_path='../../models/text_detection_ppocr/text_detection_cn_ppocrv3_2023may.onnx',
                           calibration_image_dir='../../benchmark/data/text',
                           transforms=Compose([Resize(size=(736, 736)),
                                              Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])])),
    crnn_en=Quantize(model_path='../../models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx',
                     calibration_image_dir='../../benchmark/data/text',
                     transforms=Compose([Resize(size=(100, 32)), Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]), ColorConvert(ctype=cv.COLOR_BGR2GRAY)])),
    crnn_cn=Quantize(model_path='../../models/text_recognition_crnn/text_recognition_CRNN_CN_2021nov.onnx',
                     calibration_image_dir='../../benchmark/data/text',
                     transforms=Compose([Resize(size=(100, 32))])),
    mp_palmdet=Quantize(model_path='../../models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx',
                        calibration_image_dir='path/to/dataset',
                        transforms=Compose([Resize(size=(192, 192)), Normalize(std=[255, 255, 255]),
                        ColorConvert(ctype=cv.COLOR_BGR2RGB)]), data_dim='hwc'),
    mp_handpose=Quantize(model_path='../../models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx',
                        calibration_image_dir='path/to/dataset',
                        transforms=Compose([HandAlign("mp_handpose"), Resize(size=(224, 224)), Normalize(std=[255, 255, 255]),
                        ColorConvert(ctype=cv.COLOR_BGR2RGB)]), data_dim='hwc'),
    lpd_yunet=Quantize(model_path='../../models/license_plate_detection_yunet/license_plate_detection_lpd_yunet_2023mar.onnx',
                       calibration_image_dir='../../benchmark/data/license_plate_detection',
                       transforms=Compose([Resize(size=(320, 240))]),
                       nodes_to_exclude=['MaxPool_5', 'MaxPool_18', 'MaxPool_25', 'MaxPool_32', 'MaxPool_39'],
    ),
)

if __name__ == '__main__':
    selected_models = []
    for i in range(1, len(sys.argv)):
        selected_models.append(sys.argv[i])
    if not selected_models:
        selected_models = list(models.keys())
    print('Models to be quantized: {}'.format(str(selected_models)))

    for selected_model_name in selected_models:
        q = models[selected_model_name]
        q.run()
