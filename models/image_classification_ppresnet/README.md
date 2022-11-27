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

Run the following command to try the demo:

```shell
python demo.py --input /path/to/image

# get help regarding various parameters
python demo.py --help
```

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://arxiv.org/abs/1512.03385
- https://github.com/opencv/opencv/tree/master/samples/dnn/dnn_model_runner/dnn_conversion/paddlepaddle
- https://github.com/PaddlePaddle/PaddleHub
