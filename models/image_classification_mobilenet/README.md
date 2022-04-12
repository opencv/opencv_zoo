# MobileNets

MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

MobileNetV2: Inverted Residuals and Linear Bottlenecks

Models are taken from https://github.com/shicai/MobileNet-Caffe and converted to ONNX format using [caffe2onnx](https://github.com/asiryan/caffe2onnx):
```
python -m caffe2onnx.convert --prototxt mobilenet_deploy.prototxt --caffemodel mobilenet.caffemodel --onnx mobilenet_v1.onnx
python -m caffe2onnx.convert --prototxt mobilenet_v2_deploy.prototxt --caffemodel mobilenet_v2.caffemodel --onnx mobilenet_v2.onnx
```

NOTE: Quantized MobileNet V1 & V2 have a great drop in accuracy. We are working on producing higher accuracy MobileNets.

## Demo

Run the following command to try the demo:
```shell
# MobileNet V1
python demo.py --input /path/to/image
# MobileNet V2
python demo.py --input /path/to/image --model v2
```

## License

Model weights are licensed under [BSD-3-Clause License](./LICENSE).
Scripts are licensed unser [Apache 2.0 License](../../LICENSE).

## Reference

- MobileNet V1: https://arxiv.org/abs/1704.04861
- MobileNet V2: https://arxiv.org/abs/1801.04381
- https://github.com/shicai/MobileNet-Caffe

