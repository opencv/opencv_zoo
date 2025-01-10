# Palm detector from MediaPipe Handpose

This model detects palm bounding boxes and palm landmarks, and is converted from TFLite to ONNX using following tools:

- TFLite model to ONNX: https://github.com/onnx/tensorflow-onnx
- simplified by [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)

SSD Anchors are generated from [GenMediaPipePalmDectionSSDAnchors](https://github.com/VimalMollyn/GenMediaPipePalmDectionSSDAnchors)

**Note**:
- Visit https://github.com/google/mediapipe/blob/master/docs/solutions/models.md#hands for models of larger scale.
- `palm_detection_mediapipe_2023feb_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.

## Demo

### Python

Run the following commands to try the demo:

```bash
# detect on camera input
python demo.py
# detect on an image
python demo.py -i /path/to/image -v

# get help regarding various parameters
python demo.py --help
```

### C++

Install latest OpenCV (with opencv_contrib) and CMake >= 3.24.0 to get started with:

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# detect on camera input
./build/demo
# detect on an image
./build/demo -i=/path/to/image -v
# get help messages
./build/demo -h
```

### Example outputs

![webcam demo](./example_outputs/mppalmdet_demo.gif)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- MediaPipe Handpose: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- MediaPipe hands model and model card: https://github.com/google/mediapipe/blob/master/docs/solutions/models.md#hands
- Handpose TFJS:https://github.com/tensorflow/tfjs-models/tree/master/handpose
- Int8 model quantized with rgb evaluation set of FreiHAND: https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html