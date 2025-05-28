# VIT tracker

VIT tracker(vision transformer tracker) is a much better model for real-time object tracking. VIT tracker can achieve speeds exceeding nanotrack by 20% in single-threaded mode with ARM chip, and the advantage becomes even more pronounced in multi-threaded mode. In addition, on the dataset, vit tracker demonstrates better performance compared to nanotrack. Moreover, vit trackerprovides confidence values during the tracking process, which can be used to determine if the tracking is currently lost.

In target tracking tasks, the score is an important indicator that can indicate whether the current target is lost. In the video, vit tracker can track the target and display the current score in the upper left corner of the video. When the target is lost, the score drops significantly. While nanotrack will only return 0.9 score in any situation, so that we cannot determine whether the target is lost.

Video demo: https://youtu.be/MJiPnu1ZQRI

This model is contributed by [Pengyu Liu](https://github.com/lpylpy0514) in GSoC 2023 project [**Realtime object tracking models**](https://github.com/opencv/opencv/wiki/GSoC_2023#idea-realtime-object-tracking-models)

**Note**:
- OpenCV > 4.8.0 is required. Build from source with instructions from https://opencv.org/get-started/.**
- `object_tracking_vittrack_2023sep_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.


# Demo
## Python
```bash
# tracking on camera input
python demo.py

# tracking on video
python demo.py --input /path/to/video

# get help regarding various parameters
python demo.py --help
```
## C++
Install latest OpenCV and CMake >= 3.24.0 to get started.

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# tracking on camera input
./build/opencv_zoo_object_tracking_vittrack

# tracking on video
./build/opencv_zoo_object_tracking_vittrack -i=/path/to/video

# get help messages
./build/opencv_zoo_object_tracking_vittrack -h
```

# Example outputs

<img src="example_outputs/vittrack_demo.gif" style="zoom:200%;" />


# Speed test

NOTE: The speed below is tested by **onnxruntime** because opencv has poor support for the transformer architecture for now.

ONNX speed test on ARM platform(apple M2)(ms):

| thread nums | 1    | 2    | 3    | 4             |
| ----------- | ---- | ---- | ---- | ------------- |
| nanotrack   | 5.25 | 4.86 | 4.72 | 4.49          |
| vit tracker | 4.18 | 2.41 | 1.97 | **1.46 (3X)** |

ONNX speed test on x86 platform(intel i3 10105)(ms):

| thread nums | 1    | 2    | 3    | 4    |
| ----------- | ---- | ---- | ---- | ---- |
| nanotrack   | 3.20 | 2.75 | 2.46 | 2.55 |
| vit tracker | 3.84 | 2.37 | 2.10 | 2.01 |

# Performance test

preformance test on lasot dataset(AUC is the most important data. Higher AUC means better tracker):

| LASOT       | AUC  | P    | Pnorm |
| ----------- | ---- | ---- | ----- |
| nanotrack   | 46.8 | 45.0 | 43.3  |
| vit tracker | 48.6 | 44.8 | 54.7  |

# License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

# Reference:

OSTrack: https://github.com/botaoye/OSTrack

OpenCV Sample: https://github.com/opencv/opencv/blob/4.x/samples/dnn/vit_tracker.cpp
