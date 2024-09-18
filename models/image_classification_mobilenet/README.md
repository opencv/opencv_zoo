# MobileNets

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

MobileNetV2: Inverted Residuals and Linear Bottlenecks

Results of accuracy evaluation with [tools/eval](../../tools/eval).

Quantization was done via Per Channel method for V1 and Per Tensor for V2

| Models             | Top-1 Accuracy | Top-5 Accuracy |
| ------------------ | -------------- | -------------- |
| MobileNet V1       | 67.64          | 87.97          |
| MobileNet V1 quant | 40.50          | 53.87          |
| MobileNet V2       | 69.44          | 89.23          |
| MobileNet V2 quant | 58.10          | 87.40          |

\*: 'quant' stands for 'quantized'.

## Demo

### Python

Run the following command to try the demo:

```shell
# MobileNet V1
python demo.py --input /path/to/image
# MobileNet V2
python demo.py --input /path/to/image --model v2

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
./build/opencv_zoo_image_classification_mobilenet
# detect on an image
./build/opencv_zoo_image_classification_mobilenet -m=/path/to/model -i=/path/to/image -v
# get help messages
./build/opencv_zoo_image_classification_mobilenet -h
```


## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- MobileNet V1: https://arxiv.org/abs/1704.04861
- MobileNet V2: https://arxiv.org/abs/1801.04381
- MobileNet V1 weight and scripts for training: https://github.com/wjc852456/pytorch-mobilenet-v1
- MobileNet V2 weight: https://github.com/onnx/models/tree/main/vision/classification/mobilenet
