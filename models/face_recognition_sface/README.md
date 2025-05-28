# SFace

SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition

Note:

- SFace is contributed by [Yaoyao Zhong](https://github.com/zhongyy).
- Model files encode MobileFaceNet instances trained on the SFace loss function, see the [SFace paper](https://arxiv.org/abs/2205.12010) for reference.
- ONNX file conversions from [original code base](https://github.com/zhongyy/SFace) thanks to [Chengrui Wang](https://github.com/crywang).
- (As of Sep 2021) Supporting 5-landmark warping for now, see below for details.
- `face_recognition_sface_2021dec_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.

Results of accuracy evaluation with [tools/eval](../../tools/eval).

| Models      | Accuracy |
| ----------- | -------- |
| SFace       | 0.9940   |
| SFace block | 0.9942   |
| SFace quant | 0.9932   |

\*: 'quant' stands for 'quantized'.
\*\*: 'block' stands for 'blockwise quantized'.

## Demo

***NOTE***: This demo uses [../face_detection_yunet](../face_detection_yunet) as face detector, which supports 5-landmark detection for now (2021sep).

Run the following command to try the demo:

### Python
```shell
# recognize on images
python demo.py --target /path/to/image1 --query /path/to/image2

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
./build/demo -t=/path/to/target_face
# detect on an image
./build/demo -t=/path/to/target_face -q=/path/to/query_face -v
# get help messages
./build/demo -h
```

### Example outputs

![sface demo](./example_outputs/demo.jpg)

Note: Left part of the image is the target identity, the right part is the query. Green boxes are the same identity, red boxes are different identities compared to the left.

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://ieeexplore.ieee.org/document/9318547
- https://github.com/zhongyy/SFace
