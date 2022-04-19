# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import os
import sys
import numpy as ny
import cv2 as cv

import onnx
from neural_compressor.experimental import Quantization, common as nc_Quantization, nc_common
from onnx import version_converter
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

from transform import Compose, Resize, CenterCrop, Normalize, ColorConvert

class DataReader(CalibrationDataReader):
    def __init__(self, model_path, image_dir, transforms):
        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name
        self.transforms = transforms
        self.data = self.get_calibration_data(image_dir)
        self.enum_data_dicts = iter([{self.input_name: x} for x in self.data])

    def get_next(self):
        return next(self.enum_data_dicts, None)

    def get_calibration_data(self, image_dir):
        blobs = []
        for image_name in os.listdir(image_dir):
            image_name_suffix = image_name.split('.')[-1].lower()
            if image_name_suffix != 'jpg' and image_name_suffix != 'jpeg':
                continue
            img = cv.imread(os.path.join(image_dir, image_name))
            img = self.transforms(img)
            blob = cv.dnn.blobFromImage(img)
            blobs.append(blob)
        return blobs

class ORT_Quantize:
    def __init__(self, model_path, calibration_image_dir, transforms=Compose(), per_channel=False, act_type='int8', wt_type='int8'):
        self.type_dict = {"uint8" : QuantType.QUInt8, "int8" : QuantType.QInt8}

        self.model_path = model_path
        self.calibration_image_dir = calibration_image_dir
        self.transforms = transforms
        self.per_channel = per_channel
        self.act_type = act_type
        self.wt_type = wt_type

        # data reader
        self.dr = DataReader(self.model_path, self.calibration_image_dir, self.transforms)

    def check_opset(self, convert=True):
        model = onnx.load(self.model_path)
        if model.opset_import[0].version != 11:
            print('\tmodel opset version: {}. Converting to opset 11'.format(model.opset_import[0].version))
            # convert opset version to 11
            model_opset11 = version_converter.convert_version(model, 11)
            # save converted model
            output_name = '{}-opset11.onnx'.format(self.model_path[:-5])
            onnx.save_model(model_opset11, output_name)
            # update model_path for quantization
            self.model_path = output_name

    def run(self):
        print('Quantizing {}: act_type {}, wt_type {}'.format(self.model_path, self.act_type, self.wt_type))
        self.check_opset()
        output_name = '{}-act_{}-wt_{}-quantized.onnx'.format(self.model_path[:-5], self.act_type, self.wt_type)
        quantize_static(self.model_path, output_name, self.dr,
                        per_channel=self.per_channel,
                        weight_type=self.type_dict[self.wt_type],
                        activation_type=self.type_dict[self.act_type])
        os.remove('augmented_model.onnx')
        os.remove('{}-opt.onnx'.format(self.model_path[:-5]))
        print('\tQuantized model saved to {}'.format(output_name))

class INC_Quantize:
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path

    def run(self):
        print('Quantizing (int8) with Intel\'s Neural Compressor:')
        print('\tModel: {}'.format(self.model_path))
        print('\tConfig: {}'.format(self.config_path))

        output_name = '{}-int8-quantized.onnx'.format(self.model_path[:-5])

        model = onnx.load(self.model_path)
        quantizer = nc_Quantization(self.config_path)
        quantizer.model = common.Model(model)
        q_model = quantizer()
        q_model.save(output_name)


models=dict(
    yunet=ORT_Quantize(model_path='../../models/face_detection_yunet/face_detection_yunet_2022mar.onnx',
                   calibration_image_dir='../../benchmark/data/face_detection',
                   transforms=Compose([Resize(size=(160, 120))])),
    sface=ORT_Quantize(model_path='../../models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
                   calibration_image_dir='../../benchmark/data/face_recognition',
                   transforms=Compose([Resize(size=(112, 112))])),
    pphumenseg=ORT_Quantize(model_path='../../models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2021oct.onnx',
                        calibration_image_dir='../../benchmark/data/human_segmentation',
                        transforms=Compose([Resize(size=(192, 192))])),
    ppresnet50=ORT_Quantize(model_path='../../models/image_classification_ppresnet/image_classification_ppresnet50_2022jan.onnx',
                        calibration_image_dir='../../benchmark/data/image_classification',
                        transforms=Compose([Resize(size=(224, 224))])),
    mobilenetv1=INC_Quantize(model_path='../../models/image_classification_mobilenet/image_classification_mobilenetv1_2022apr.onnx',
                             config_path='./inc_configs/mobilenet.yaml'),
    mobilenetv2=INC_Quantize(model_path='../../models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx',
                             config_path='./inc_configs/mobilenet.yaml'),
    # TBD: DaSiamRPN
    youtureid=ORT_Quantize(model_path='../../models/person_reid_youtureid/person_reid_youtu_2021nov.onnx',
                       calibration_image_dir='../../benchmark/data/person_reid',
                       transforms=Compose([Resize(size=(128, 256))])),
    # TBD: DB-EN & DB-CN
    crnn_en=ORT_Quantize(model_path='../../models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx',
                     calibration_image_dir='../../benchmark/data/text',
                     transforms=Compose([Resize(size=(100, 32)), ColorConvert(ctype=cv.COLOR_BGR2GRAY)])),
    crnn_cn=ORT_Quantize(model_path='../../models/text_recognition_crnn/text_recognition_CRNN_CN_2021nov.onnx',
                     calibration_image_dir='../../benchmark/data/text',
                     transforms=Compose([Resize(size=(100, 32))]))
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

