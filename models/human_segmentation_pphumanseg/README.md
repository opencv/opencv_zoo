# PPHumanSeg

This model is ported from [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) using [this script from OpenCV](https://github.com/opencv/opencv/blob/master/samples/dnn/dnn_model_runner/dnn_conversion/paddlepaddle/paddle_humanseg.py).

**Note**:
- `human_segmentation_pphumanseg_2023mar_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.

## Demo

### Python

Run the following command to try the demo:

```shell
# detect on camera input
python demo.py
# detect on an image
python demo.py --input /path/to/image -v

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
./build/opencv_zoo_human_segmentation
# detect on an image
./build/opencv_zoo_human_segmentation -i=/path/to/image
# get help messages
./build/opencv_zoo_human_segmentation -h
```

### Example outputs

![webcam demo](./example_outputs/pphumanseg_demo.gif)

![messi](./example_outputs/messi.jpg)

---
Results of accuracy evaluation with [tools/eval](../../tools/eval).

| Models             | Accuracy       | mIoU          |
| ------------------ | -------------- | ------------- |
| PPHumanSeg         | 0.9656         | 0.9164        |
| PPHumanSeg block   | 0.9655         | 0.9162        |
| PPHumanSeg quant   | 0.7285         | 0.3642        |


\*: 'quant' stands for 'quantized'.
\*\*: 'block' stands for 'blockwise quantized'.

---
## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://arxiv.org/abs/1512.03385
- https://github.com/opencv/opencv/tree/master/samples/dnn/dnn_model_runner/dnn_conversion/paddlepaddle
- https://github.com/PaddlePaddle/PaddleHub
