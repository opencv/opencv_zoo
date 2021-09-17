import os
import argparse

import yaml
import tqdm
import numpy as np
import cv2 as cv

from models import MODELS
from download import Downloader

parser = argparse.ArgumentParser("Benchmarks for OpenCV Zoo.")
parser.add_argument('--cfg', '-c', type=str,
                    help='Benchmarking on the given config.')
args = parser.parse_args()

class Timer:
    def __init__(self):
        self._tm = cv.TickMeter()

        self._time_record = []
        self._average_time = 0
        self._calls = 0

    def start(self):
        self._tm.start()

    def stop(self):
        self._tm.stop()
        self._calls += 1
        self._time_record.append(self._tm.getTimeMilli())
        self._average_time = sum(self._time_record) / self._calls
        self._tm.reset()

    def reset(self):
        self._time_record = []
        self._average_time = 0
        self._calls = 0

    def getAverageTime(self):
        return self._average_time


class Benchmark:
    def __init__(self, **kwargs):
        self._fileList = kwargs.pop('fileList', None)
        assert self._fileList, 'fileList cannot be empty'

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

        self._sizes = kwargs.pop('sizes', None)
        self._repeat = kwargs.pop('repeat', 100)
        self._parentPath = kwargs.pop('parentPath', 'benchmark/data')
        self._useGroundTruth = kwargs.pop('useDetectionLabel', False) # If it is enable, 'sizes' will not work
        assert (self._sizes and not self._useGroundTruth) or (not self._sizes and self._useGroundTruth), 'If \'useDetectionLabel\' is True, \'sizes\' should not exist.'

        self._timer = Timer()
        self._benchmark_results = dict.fromkeys(self._fileList, dict())

        if self._useGroundTruth:
            self.loadLabel()

    def loadLabel(self):
        self._labels = dict.fromkeys(self._fileList, None)
        for imgName in self._fileList:
            self._labels[imgName] = np.loadtxt(os.path.join(self._parentPath, '{}.txt'.format(imgName[:-4])))

    def run(self, model):
        model.setBackend(self._backend)
        model.setTarget(self._target)

        for imgName in self._fileList:
            img = cv.imread(os.path.join(self._parentPath, imgName))
            if self._useGroundTruth:
                for idx, gt in enumerate(self._labels[imgName]):
                    self._benchmark_results[imgName]['gt{}'.format(idx)] = self._run(
                        model,
                        img,
                        gt,
                        pbar_msg='  {}, gt{}'.format(imgName, idx)
                    )
            else:
                if self._sizes is None:
                    h, w, _ = img.shape
                    model.setInputSize([w, h])
                    self._benchmark_results[imgName][str([w, h])] = self._run(
                        model,
                        img,
                        pbar_msg='  {}, original size {}'.format(imgName, str([w, h]))
                    )
                else:
                    for size in self._sizes:
                        imgResized = cv.resize(img, size)
                        model.setInputSize(size)
                        self._benchmark_results[imgName][str(size)] = self._run(
                            model,
                            imgResized,
                            pbar_msg='  {}, size {}'.format(imgName, str(size))
                        )

    def printResults(self):
        print('  Results:')
        for imgName, results in self._benchmark_results.items():
            print('    image: {}'.format(imgName))
            total_latency = 0
            for key, latency in results.items():
                total_latency += latency
                print('        {}, latency: {:.4f} ms'.format(key, latency))
            print('        Average latency: {:.4f} ms'.format(total_latency / len(results)))

    def _run(self, model, *args, **kwargs):
        self._timer.reset()
        pbar = tqdm.tqdm(range(self._repeat))
        for _ in pbar:
            pbar.set_description(kwargs.get('pbar_msg', None))

            self._timer.start()
            results = model.infer(*args)
            self._timer.stop()
        return self._timer.getAverageTime()


def build_from_cfg(cfg, registery):
    obj_name = cfg.pop('name')
    obj = registery.get(obj_name)
    return obj(**cfg)

def prepend_pythonpath(cfg, key1, key2):
    pythonpath = os.environ['PYTHONPATH']
    if cfg[key1][key2].startswith('/'):
        return
    cfg[key1][key2] = os.path.join(pythonpath, cfg[key1][key2])

if __name__ == '__main__':
    assert args.cfg.endswith('yaml'), 'Currently support configs of yaml format only.'
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # prepend PYTHONPATH to each path
    prepend_pythonpath(cfg, key1='Data', key2='parentPath')
    prepend_pythonpath(cfg, key1='Benchmark', key2='parentPath')
    prepend_pythonpath(cfg, key1='Model', key2='modelPath')


    # Download data if not exist
    print('Loading data:')
    downloader = Downloader(**cfg['Data'])
    downloader.get()

    # Instantiate benchmarking
    benchmark = Benchmark(**cfg['Benchmark'])

    # Instantiate model
    model = build_from_cfg(cfg=cfg['Model'], registery=MODELS)

    # Run benchmarking
    print('Benchmarking {}:'.format(model.name))
    benchmark.run(model)
    benchmark.printResults()