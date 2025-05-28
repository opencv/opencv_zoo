# Person detector from MediaPipe Pose

This model detects upper body and full body keypoints of a person, and is downloaded from https://github.com/PINTO0309/PINTO_model_zoo/blob/main/053_BlazePose/20_densify_pose_detection/download.sh or converted from TFLite to ONNX using following tools:

- TFLite model to ONNX with MediaPipe custom `densify` op: https://github.com/PINTO0309/tflite2tensorflow
- simplified by [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)

SSD Anchors are generated from [GenMediaPipePalmDectionSSDAnchors](https://github.com/VimalMollyn/GenMediaPipePalmDectionSSDAnchors)

**Note**:
- `person_detection_mediapipe_2023mar_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.

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

Install latest OpenCV and CMake >= 3.24.0 to get started with:

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# detect on camera input
./build/opencv_zoo_person_detection_mediapipe
# detect on an image
./build/opencv_zoo_person_detection_mediapipe -m=/path/to/model -i=/path/to/image -v
# get help messages
./build/opencv_zoo_person_detection_mediapipe -h
```

### Example outputs

![webcam demo](./example_outputs/mppersondet_demo.webp)

## License

All files in this directory are licensed under [Apache 2.0 License](LICENSE).

## Reference
- MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- MediaPipe pose model and model card: https://github.com/google/mediapipe/blob/master/docs/solutions/models.md#pose
- BlazePose TFJS: https://github.com/tensorflow/tfjs-models/tree/master/pose-detection/src/blazepose_tfjs
