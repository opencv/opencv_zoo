# SFace

SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition

Note:
- SFace is contributed by [Yaoyao Zhong](https://github.com/zhongyy/SFace).
- [face_recognition_sface_2021sep.onnx](./face_recognition_sface_2021sep.onnx) is converted from the model from https://github.com/zhongyy/SFace thanks to [Chengrui Wang](https://github.com/crywang).
- Support 5-landmark warpping for now (2021sep)

## Demo

***NOTE***: This demo uses [../face_detection_yunet](../face_detection_yunet) as face detector, which supports 5-landmark detection for now (2021sep).

Run the following command to try the demo:
```shell
# recognize on images
python demo.py --input1 /path/to/image1 --input2 /path/to/image2
```

## Evaluation
We used the accuracy evaluation tools from [opencv_zoo/tools/eval](../../tools/eval).

### LFW
Performance on [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/):
```shell
Validation rate: 0.34067+-0.01384 @ FAR=0.00100
acc=0.8483, std=0.0138
```



## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://ieeexplore.ieee.org/document/9318547
- https://github.com/zhongyy/SFace