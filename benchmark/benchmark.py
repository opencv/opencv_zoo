import os
import argparse

import yaml
import numpy as np
import cv2 as cv

from models import MODELS
from utils import METRICS, DATALOADERS

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]
backend_target_str_pairs = [
    ["cv.dnn.DNN_BACKEND_OPENCV", "cv.dnn.DNN_TARGET_CPU"],
    ["cv.dnn.DNN_BACKEND_CUDA",   "cv.dnn.DNN_TARGET_CUDA"],
    ["cv.dnn.DNN_BACKEND_CUDA",   "cv.dnn.DNN_TARGET_CUDA_FP16"],
    ["cv.dnn.DNN_BACKEND_TIMVX",  "cv.dnn.DNN_TARGET_NPU"],
    ["cv.dnn.DNN_BACKEND_CANN",   "cv.dnn.DNN_TARGET_NPU"]
]

parser = argparse.ArgumentParser("Benchmarks for OpenCV Zoo.")
parser.add_argument('--cfg', '-c', type=str,
                    help='Benchmarking on the given config.')
parser.add_argument('--cfg_overwrite_backend_target', type=int, default=-1,
                    help='''Choose one of the backend-target pair to run this demo:
                        others: (default) use the one from config,
                        {:d}: OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument("--cfg_exclude", type=str, help="Configs to be excluded when using --all. Split keywords with colons (:). Not sensitive to upper/lower case.")
parser.add_argument("--model_exclude", type=str, help="Models to be excluded. Split model names with colons (:). Sensitive to upper/lower case.")
parser.add_argument("--fp32", action="store_true", help="Benchmark models of float32 precision only.")
parser.add_argument("--fp16", action="store_true", help="Benchmark models of float16 precision only.")
parser.add_argument("--int8", action="store_true", help="Benchmark models of int8 precision only.")
parser.add_argument("--int8bq", action="store_true", help="Benchmark models of blocked int8 precision only.")
parser.add_argument("--all", action="store_true", help="Benchmark all models")
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
            timvx=cv.dnn.DNN_BACKEND_TIMVX,
            cann=cv.dnn.DNN_BACKEND_CANN,
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
            npu=cv.dnn.DNN_TARGET_NPU,
        )

        self._backend = available_backends[backend_id]
        self._target = available_targets[target_id]

        self._benchmark_results = dict()
        self._benchmark_results_brief = dict()

    def setBackendAndTarget(self, backend_id, target_id):
        self._backend = backend_id
        self._target = target_id

    def run(self, model):
        model.setBackendAndTarget(self._backend, self._target)

        for idx, data in enumerate(self._dataloader):
            filename, input_data = data[:2]

            if isinstance(input_data, np.ndarray):
                size = [input_data.shape[1], input_data.shape[0]]
            else:
                size = input_data.getFrameSize()

            if str(size) not in self._benchmark_results:
                self._benchmark_results[str(size)] = dict()
            self._benchmark_results[str(size)][filename] = self._metric.forward(model, *data[1:])

            if str(size) not in self._benchmark_results_brief:
                self._benchmark_results_brief[str(size)] = []
            self._benchmark_results_brief[str(size)] += self._benchmark_results[str(size)][filename]

    def printResults(self, model_name, model_path):
        for imgSize, res in self._benchmark_results_brief.items():
            mean, median, minimum = self._metric.getPerfStats(res)
            print("{:<10.2f} {:<10.2f} {:<10.2f} {:<12} {} with {}".format(
                mean, median, minimum, imgSize, model_name, model_path
            ))

if __name__ == '__main__':
    cfgs = []
    if args.cfg is not None:
        assert args.cfg.endswith('yaml'), 'Currently support configs of yaml format only.'
        with open(args.cfg, 'r') as f:
            cfg = yaml.safe_load(f)
        cfgs.append(cfg)
    elif args.all:
        excludes = []
        if args.cfg_exclude is not None:
            excludes = args.cfg_exclude.split(":")

        for cfg_fname in sorted(os.listdir("config")):
            skip_flag = False
            for exc in excludes:
                if exc.lower() in cfg_fname.lower():
                    skip_flag = True
            if skip_flag:
                # print("{} is skipped.".format(cfg_fname))
                continue

            assert cfg_fname.endswith("yaml"), "Currently support yaml configs only."
            with open(os.path.join("config", cfg_fname), "r") as f:
                cfg = yaml.safe_load(f)
            cfgs.append(cfg)
    else:
        raise NotImplementedError("Specify either one config or use flag --all for benchmark.")

    print("Benchmarking ...")
    if args.all:
        backend_target_id = args.cfg_overwrite_backend_target if args.cfg_overwrite_backend_target >= 0 else 0
        backend_str = backend_target_str_pairs[backend_target_id][0]
        target_str = backend_target_str_pairs[backend_target_id][1]
        print("backend={}".format(backend_str))
        print("target={}".format(target_str))
    print("{:<10} {:<10} {:<10} {:<12} {}".format("mean", "median", "min", "input size", "model"))
    for cfg in cfgs:
        # Instantiate benchmark
        benchmark = Benchmark(**cfg['Benchmark'])

        # Set backend and target
        if args.cfg_overwrite_backend_target >= 0:
            backend_id = backend_target_pairs[args.cfg_overwrite_backend_target][0]
            target_id = backend_target_pairs[args.cfg_overwrite_backend_target][1]
            benchmark.setBackendAndTarget(backend_id, target_id)

        # Instantiate model
        model_config = cfg['Model']
        model_handler, model_paths = MODELS.get(model_config.pop('name'))

        _model_paths = []
        if args.fp32 or args.fp16 or args.int8 or args.int8bq:
            if args.fp32:
                _model_paths += model_paths['fp32']
            if args.fp16:
                _model_paths += model_paths['fp16']
            if args.int8:
                _model_paths += model_paths['int8']
            if args.int8bq:
                _model_paths += model_paths['int8bq']
        else:
            _model_paths = model_paths['fp32'] + model_paths['fp16'] + model_paths['int8'] + model_paths["int8bq"]
        # filter out excluded models
        excludes = []
        if args.model_exclude is not None:
            excludes = args.model_exclude.split(":")
        _model_paths_excluded = []
        for model_path in _model_paths:
            skip_flag = False
            for mp in model_path:
                for exc in excludes:
                    if exc in mp:
                        skip_flag = True
            if skip_flag:
                continue
            _model_paths_excluded.append(model_path)
        _model_paths = _model_paths_excluded

        for model_path in _model_paths:
            model = model_handler(*model_path, **model_config)
            # Format model_path
            for i in range(len(model_path)):
                model_path[i] = model_path[i].split('/')[-1]
            # Run benchmark
            benchmark.run(model)
            benchmark.printResults(model.name, model_path)
