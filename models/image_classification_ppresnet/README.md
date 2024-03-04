# ResNet

Deep Residual Learning for Image Recognition

This model is ported from [PaddleHub](https://github.com/PaddlePaddle/PaddleHub) using [this script from OpenCV](https://github.com/opencv/opencv/blob/master/samples/dnn/dnn_model_runner/dnn_conversion/paddlepaddle/paddle_resnet50.py).

Results of accuracy evaluation with [tools/eval](../../tools/eval).

| Models          | Top-1 Accuracy | Top-5 Accuracy |
| --------------- | -------------- | -------------- |
| PP-ResNet       | 82.28          | 96.15          |
| PP-ResNet quant | 0.22           | 0.96           |

\*: 'quant' stands for 'quantized'.

## Demo

Run the following commands to try the demo:

### Python

```shell
python demo.py --input /path/to/image

# get help regarding various parameters
python demo.py --help
```
### C++

Install latest OpenCV and CMake >= 3.24.0 to get started with:

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# detect on an image
./build/opencv_zoo_image_classification_ppresnet -i=/path/to/image

# detect on an image and display top N classes
./build/opencv_zoo_image_classification_ppresnet -i=/path/to/image -k=N

# get help messages
./build/opencv_zoo_image_classification_ppresnet -h
```

### Example Results
![fish](./example_input/goldfish.jpg)
```
Predicted Top-K Labels (in decreasing confidence): 
(1) goldfish
(2) anemone fish
(3) rock beauty
(4) coral reef
(5) tench
```

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://arxiv.org/abs/1512.03385
- https://github.com/opencv/opencv/tree/master/samples/dnn/dnn_model_runner/dnn_conversion/paddlepaddle
- https://github.com/PaddlePaddle/PaddleHub
