import os
import argparse

import yaml
import numpy as np
import cv2 as cv

# from ..models import MODELS
from models import MODELS
from utils import METRICS

parser = argparse.ArgumentParser("Benchmarks for OpenCV Zoo.")
parser.add_argument('--cfg', '-c', type=str,
                    help='Benchmarking on the given config.')
args = parser.parse_args()

def build_from_cfg(cfg, registery, key='name'):
    obj_name = cfg.pop(key)
    obj = registery.get(obj_name)
    return obj(**cfg)

def prepend_pythonpath(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            prepend_pythonpath(v)
        else:
            if 'path' in k.lower():
                cfg[k] = os.path.join(os.environ['PYTHONPATH'], v)

class Data:
    def __init__(self, **kwargs):
        self._path = kwargs.pop('path', None)
        assert self._path, 'Benchmark[\'data\'][\'path\'] cannot be empty.'

        self._files = kwargs.pop('files', None)
        if not self._files:
            print('Benchmark[\'data\'][\'files\'] is empty, loading all images by default.')
            self._files = list()
            for filename in os.listdir(self._path):
                if filename.endswith('jpg') or filename.endswith('png'):
                    self._files.append(filename)

        self._use_label = kwargs.pop('useLabel', False)
        if self._use_label:
            self._labels = self._load_label()

        self._to_rgb = kwargs.pop('toRGB', False)
        self._resize = tuple(kwargs.pop('resize', []))
        self._center_crop = kwargs.pop('centerCrop', None)

    def _load_label(self):
        labels = dict.fromkeys(self._files, None)
        for filename in self._files:
            labels[filename] = np.loadtxt(os.path.join(self._path, '{}.txt'.format(filename[:-4])), ndmin=2)
        return labels

    def __getitem__(self, idx):
        image = cv.imread(os.path.join(self._path, self._files[idx]))

        if self._to_rgb:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self._resize:
            image = cv.resize(image, self._resize)
        if self._center_crop:
            h, w, _ = image.shape
            w_crop = int((w - self._center_crop) / 2.)
            assert w_crop >= 0
            h_crop = int((h - self._center_crop) / 2.)
            assert h_crop >= 0
            image = image[w_crop:w-w_crop, h_crop:h-h_crop, :]

        if self._use_label:
            return self._files[idx], image, self._labels[self._files[idx]]
        else:
            return self._files[idx], image

class Benchmark:
    def __init__(self, **kwargs):
        self._data_dict = kwargs.pop('data', None)
        assert self._data_dict, 'Benchmark[\'data\'] cannot be empty and must have path and files.'
        self._data = Data(**self._data_dict)

        self._metric_dict = kwargs.pop('metric', None)
        # self._metric = Metric(**self._metric_dict)
        self._metric = build_from_cfg(self._metric_dict, registery=METRICS, key='type')

        backend_id = kwargs.pop('backend', 'default')
        available_backends = dict(
            default=cv.dnn.DNN_BACKEND_DEFAULT,
            # halide=cv.dnn.DNN_BACKEND_HALIDE,
            # inference_engine=cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
            opencv=cv.dnn.DNN_BACKEND_OPENCV,
            # vkcom=cv.dnn.DNN_BACKEND_VKCOM,
            cuda=cv.dnn.DNN_BACKEND_CUDA
        )
        self._backend = available_backends[backend_id]

        target_id = kwargs.pop('target', 'cpu')
        available_targets = dict(
            cpu=cv.dnn.DNN_TARGET_CPU,
            # opencl=cv.dnn.DNN_TARGET_OPENCL,
            # opencl_fp16=cv.dnn.DNN_TARGET_OPENCL_FP16,
            # myriad=cv.dnn.DNN_TARGET_MYRIAD,
            # vulkan=cv.dnn.DNN_TARGET_VULKAN,
            # fpga=cv.dnn.DNN_TARGET_FPGA,
            cuda=cv.dnn.DNN_TARGET_CUDA,
            cuda_fp16=cv.dnn.DNN_TARGET_CUDA_FP16,
            # hddl=cv.dnn.DNN_TARGET_HDDL
        )
        self._target = available_targets[target_id]

        self._benchmark_results = dict()

    def run(self, model):
        model.setBackend(self._backend)
        model.setTarget(self._target)

        for data in self._data:
            self._benchmark_results[data[0]] = self._metric.forward(model, *data[1:])

    def printResults(self):
        for imgName, results in self._benchmark_results.items():
            print('  image: {}'.format(imgName))
            total_latency = 0
            for key, latency in results.items():
                total_latency += latency
                print('      {}, latency ({}): {:.4f} ms'.format(key, self._metric.getReduction(), latency))

if __name__ == '__main__':
    assert args.cfg.endswith('yaml'), 'Currently support configs of yaml format only.'
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # prepend PYTHONPATH to each path
    prepend_pythonpath(cfg)

    # Instantiate benchmarking
    benchmark = Benchmark(**cfg['Benchmark'])

    # Instantiate model
    model = build_from_cfg(cfg=cfg['Model'], registery=MODELS)

    # Run benchmarking
    print('Benchmarking {}:'.format(model.name))
    benchmark.run(model)
    benchmark.printResults()