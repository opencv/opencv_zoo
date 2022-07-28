# SFace

SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition

Note:
- SFace is contributed by [Yaoyao Zhong](https://github.com/zhongyy/SFace).
- [face_recognition_sface_2021sep.onnx](./face_recognition_sface_2021sep.onnx) is converted from the model from https://github.com/zhongyy/SFace thanks to [Chengrui Wang](https://github.com/crywang).
- Support 5-landmark warpping for now (2021sep)

Results of accuracy evaluation with [tools/eval](../../tools/eval).

| Models      | Accuracy | 
|-------------|----------|
| SFace       | 0.9940   |
| SFace quant | 0.9932   |

\*: 'quant' stands for 'quantized'.


## Demo

***NOTE***: This demo uses [../face_detection_yunet](../face_detection_yunet) as face detector, which supports 5-landmark detection for now (2021sep).

Run the following command to try the demo:
```shell
# recognize on images
python demo.py --input1 /path/to/image1 --input2 /path/to/image2
```


## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://ieeexplore.ieee.org/document/9318547
- https://github.com/zhongyy/SFace