import os
import sys
import numpy as np
import cv2 as cv

import onnx
from neural_compressor.experimental import Quantization, common

class Quantize:
    def __init__(self, model_path, config_path, custom_dataset=None):
        self.model_path = model_path
        self.config_path = config_path
        self.custom_dataset = custom_dataset

    def run(self):
        print('Quantizing (int8) with Intel\'s Neural Compressor:')
        print('\tModel: {}'.format(self.model_path))
        print('\tConfig: {}'.format(self.config_path))

        output_name = '{}-int8-quantized.onnx'.format(self.model_path[:-5])

        model = onnx.load(self.model_path)
        quantizer = Quantization(self.config_path)
        if self.custom_dataset is not None:
            quantizer.calib_dataloader = common.DataLoader(self.custom_dataset)
        quantizer.model = common.Model(model)
        q_model = quantizer()
        q_model.save(output_name)

class Dataset:
    def __init__(self, root, size=None, toTensor=False):
        self.root = root
        self.size = size
        self.toTensor = toTensor

        self.image_list = self.load_image_list(self.root)

    def load_image_list(self, path):
        image_list = []
        for f in os.listdir(path):
            if not f.endswith('.jpg'):
                continue
            image_list.append(os.path.join(path, f))
        return image_list

    def __getitem__(self, idx):
        img = cv.imread(self.image_list[idx])
        if self.size:
            img = cv.resize(img, dsize=self.size)
        if self.toTensor:
            img = img.transpose(2, 0, 1) # hwc -> chw
            img = img.astype(np.float32)
        return img, 1

    def __len__(self):
        return len(self.image_list)

models=dict(
    mobilenetv1=Quantize(model_path='../../models/image_classification_mobilenet/image_classification_mobilenetv1_2022apr.onnx',
                             config_path='./inc_configs/mobilenet.yaml'),
    mobilenetv2=Quantize(model_path='../../models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx',
                             config_path='./inc_configs/mobilenet.yaml'),
    mppalm_det=Quantize(model_path='../../models/palm_detection_mediapipe/palm_detection_mediapipe_2022may.onnx',
                             config_path='./inc_configs/mppalmdet.yaml',
                             custom_dataset=Dataset(root='../../benchmark/data/palm_detection')),
    lpd_yunet=Quantize(model_path='../../models/license_plate_detection_yunet/license_plate_detection_lpd_yunet_2022may.onnx',
                       config_path='./inc_configs/lpd_yunet.yaml',
                       custom_dataset=Dataset(root='../../benchmark/data/license_plate_detection', size=(320, 240), toTensor=True)),
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

