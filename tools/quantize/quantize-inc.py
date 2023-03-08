import os
import sys
import numpy as np
import cv2 as cv

import onnx
from neural_compressor.experimental import Quantization, common
from neural_compressor.experimental.metric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, *args):
        self.pred_list = []
        self.label_list = []
        self.samples = 0

    def update(self, predict, label):
        predict = np.array(predict)
        label = np.array(label)
        self.pred_list.append(np.argmax(predict[0]))
        self.label_list.append(label[0][0])
        self.samples += 1

    def reset(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0

    def result(self):
        correct_num = np.sum(np.array(self.pred_list) == np.array(self.label_list))
        return correct_num / self.samples


class Quantize:
    def __init__(self, model_path, config_path, custom_dataset=None, eval_dataset=None, metric=None):
        self.model_path = model_path
        self.config_path = config_path
        self.custom_dataset = custom_dataset
        self.eval_dataset = eval_dataset
        self.metric = metric

    def run(self):
        print('Quantizing (int8) with Intel\'s Neural Compressor:')
        print('\tModel: {}'.format(self.model_path))
        print('\tConfig: {}'.format(self.config_path))

        output_name = '{}-int8-quantized.onnx'.format(self.model_path[:-5])

        model = onnx.load(self.model_path)
        quantizer = Quantization(self.config_path)
        quantizer.model = common.Model(model)
        if self.custom_dataset is not None:
            quantizer.calib_dataloader = common.DataLoader(self.custom_dataset)
        if self.eval_dataset is not None:
            quantizer.eval_dataloader = common.DataLoader(self.eval_dataset)
        if self.metric is not None:
            quantizer.metric = common.Metric(metric_cls=self.metric, name='metric')
        q_model = quantizer()
        q_model.save(output_name)


class Dataset:
    def __init__(self, root, size=None, dim='chw', scale=1.0, mean=0.0, std=1.0, swapRB=False, toFP32=False):
        self.root = root
        self.size = size
        self.dim = dim
        self.scale = scale
        self.mean = mean
        self.std = std
        self.swapRB = swapRB
        self.toFP32 = toFP32

        self.image_list, self.label_list = self.load_image_list(self.root)

    def load_image_list(self, path):
        image_list = []
        label_list = []
        for f in os.listdir(path):
            if not f.endswith('.jpg'):
                continue
            image_list.append(os.path.join(path, f))
            label_list.append(1)
        return image_list, label_list

    def __getitem__(self, idx):
        img = cv.imread(self.image_list[idx])

        if self.swapRB:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.size:
            img = cv.resize(img, dsize=self.size)

        if self.toFP32:
            img = img.astype(np.float32)

        img = img * self.scale
        img = img - self.mean
        img = img / self.std

        if self.dim == 'chw':
            img = img.transpose(2, 0, 1)  # hwc -> chw

        return img, self.label_list[idx]

    def __len__(self):
        return len(self.image_list)


class FerDataset(Dataset):
    def __init__(self, root, size=None, dim='chw', scale=1.0, mean=0.0, std=1.0, swapRB=False, toFP32=False):
        super(FerDataset, self).__init__(root, size, dim, scale, mean, std, swapRB, toFP32)

    def load_image_list(self, path):
        image_list = []
        label_list = []
        for f in os.listdir(path):
            if not f.endswith('.jpg'):
                continue
            image_list.append(os.path.join(path, f))
            label_list.append(int(f.split("_")[2]))
        return image_list, label_list


models = dict(
    mobilenetv1=Quantize(model_path='../../models/image_classification_mobilenet/image_classification_mobilenetv1_2022apr.onnx',
                         config_path='./inc_configs/mobilenet.yaml'),
    mobilenetv2=Quantize(model_path='../../models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx',
                         config_path='./inc_configs/mobilenet.yaml'),
    mp_handpose=Quantize(model_path='../../models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2022may.onnx',
                         config_path='./inc_configs/mp_handpose.yaml',
                         custom_dataset=Dataset(root='../../benchmark/data/palm_detection', dim='hwc', swapRB=True, mean=127.5, std=127.5, toFP32=True)),
    fer=Quantize(model_path='../../models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx',
                 config_path='./inc_configs/fer.yaml',
                 custom_dataset=FerDataset(root='../../benchmark/data/facial_expression_recognition/fer_calibration', size=(112, 112), toFP32=True, swapRB=True, scale=1./255, mean=0.5, std=0.5),
                 eval_dataset=FerDataset(root='../../benchmark/data/facial_expression_recognition/fer_evaluation', size=(112, 112), toFP32=True, swapRB=True, scale=1./255, mean=0.5, std=0.5),
                 metric=Accuracy),
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
