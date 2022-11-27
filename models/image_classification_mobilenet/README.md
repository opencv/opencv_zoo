# MobileNets

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

MobileNetV2: Inverted Residuals and Linear Bottlenecks

Results of accuracy evaluation with [tools/eval](../../tools/eval).

| Models             | Top-1 Accuracy | Top-5 Accuracy |
| ------------------ | -------------- | -------------- |
| MobileNet V1       | 67.64          | 87.97          |
| MobileNet V1 quant | 55.53          | 78.74          |
| MobileNet V2       | 69.44          | 89.23          |
| MobileNet V2 quant | 68.37          | 88.56          |

\*: 'quant' stands for 'quantized'.

## Demo

Run the following command to try the demo:

```shell
# MobileNet V1
python demo.py --input /path/to/image
# MobileNet V2
python demo.py --input /path/to/image --model v2

# get help regarding various parameters
python demo.py --help
```

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- MobileNet V1: https://arxiv.org/abs/1704.04861
- MobileNet V2: https://arxiv.org/abs/1801.04381
- MobileNet V1 weight and scripts for training: https://github.com/wjc852456/pytorch-mobilenet-v1
- MobileNet V2 weight: https://github.com/onnx/models/tree/main/vision/classification/mobilenet
