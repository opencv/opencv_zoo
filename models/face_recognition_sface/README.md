# SFace

SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition

Note:

- SFace is contributed by [Yaoyao Zhong](https://github.com/zhongyy).
- Model files encode MobileFaceNet instances trained on the SFace loss function, see the [SFace paper](https://arxiv.org/abs/2205.12010) for reference.
- ONNX file conversions from [original code base](https://github.com/zhongyy/SFace) thanks to [Chengrui Wang](https://github.com/crywang).
- (As of Sep 2021) Supporting 5-landmark warping for now, see below for details.

Results of accuracy evaluation with [tools/eval](../../tools/eval).

| Models      | Accuracy |
| ----------- | -------- |
| SFace       | 0.9940   |
| SFace quant | 0.9932   |

\*: 'quant' stands for 'quantized'.

## Demo

***NOTE***: This demo uses [../face_detection_yunet](../face_detection_yunet) as face detector, which supports 5-landmark detection for now (2021sep).

Run the following command to try the demo:

```shell
# recognize on images
python demo.py --input1 /path/to/image1 --input2 /path/to/image2

# get help regarding various parameters
python demo.py --help
```

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://ieeexplore.ieee.org/document/9318547
- https://github.com/zhongyy/SFace
