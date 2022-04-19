import os
import argparse

import yaml
import numpy as np
import cv2 as cv

from models import MODELS
from utils import METRICS, DATALOADERS

parser = argparse.ArgumentParser("Benchmarks for OpenCV Zoo.")
parser.add_argument('--cfg', '-c', type=str,
                    help='Benchmarking on the given config.')
args = parser.parse_args()

def build_from_cfg(cfg, registery, key=None, name=None):
    if key is not None:
        obj_name = cfg.pop(key)
        obj = registery.get(obj_name)
        return obj(**cfg)
    elif name is not None:
        obj = registery.get(name)
        return obj(**cfg)
    else:
        raise NotImplementedError()

def prepend_pythonpath(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            prepend_pythonpath(v)
        else:
            if 'path' in k.lower():
                cfg[k] = os.path.join(os.environ['PYTHONPATH'], v)

class Benchmark:
    def __init__(self, **kwargs):
        self._type = kwargs.pop('type', None)
        if self._type is None:
            self._type = 'Base'
            print('Benchmark[\'type\'] is omitted, set to \'Base\' by default.')

        self._data_dict = kwargs.pop('data', None)
        assert self._data_dict, 'Benchmark[\'data\'] cannot be empty and must have path and files.'
        if 'type' in self._data_dict:
            self._dataloader = build_from_cfg(self._data_dict, registery=DATALOADERS, key='type')
        else:
            self._dataloader = build_from_cfg(self._data_dict, registery=DATALOADERS, name=self._type)

        self._metric_dict = kwargs.pop('metric', None)
        assert self._metric_dict, 'Benchmark[\'metric\'] cannot be empty.'
        if 'type' in self._metric_dict:
            self._metric = build_from_cfg(self._metric_dict, registery=METRICS, key='type')
        else:
            self._metric = build_from_cfg(self._metric_dict, registery=METRICS, name=self._type)

        backend_id = kwargs.pop('backend', 'default')
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
            # opencl=cv.dnn.DNN_TARGET_OPENCL,
            # opencl_fp16=cv.dnn.DNN_TARGET_OPENCL_FP16,
            # myriad=cv.dnn.DNN_TARGET_MYRIAD,
            # vulkan=cv.dnn.DNN_TARGET_VULKAN,
            # fpga=cv.dnn.DNN_TARGET_FPGA,
            cuda=cv.dnn.DNN_TARGET_CUDA,
            cuda_fp16=cv.dnn.DNN_TARGET_CUDA_FP16,
            # hddl=cv.dnn.DNN_TARGET_HDDL,
        )

        # add extra backends & targets
        try:
            available_backends['timvx'] = cv.dnn.DNN_BACKEND_TIMVX
            available_targets['npu'] = cv.dnn.DNN_TARGET_NPU
        except:
            print('OpenCV is not compiled with TIM-VX backend enbaled. See https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU for more details on how to enable TIM-VX backend.')

        self._backend = available_backends[backend_id]
        self._target = available_targets[target_id]

        self._benchmark_results = dict()

    def run(self, model):
        model.setBackend(self._backend)
        model.setTarget(self._target)

        for idx, data in enumerate(self._dataloader):
            filename, input_data = data[:2]
            if filename not in self._benchmark_results:
                self._benchmark_results[filename] = dict()
            if isinstance(input_data, np.ndarray):
                size = [input_data.shape[1], input_data.shape[0]]
            else:
                size = input_data.getFrameSize()
            self._benchmark_results[filename][str(size)] = self._metric.forward(model, *data[1:])

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
    model = build_from_cfg(cfg=cfg['Model'], registery=MODELS, key='name')

    # Run benchmarking
    print('Benchmarking {}:'.format(model.name))
    benchmark.run(model)
    benchmark.printResults()
