# SFace

SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition

SFace is contributed by [Yaoyao Zhong](https://github.com/zhongyy/SFace). [face_recognition_sface.onnx](./face_recognition_sface.onnx) is converted from the model from https://github.com/zhongyy/SFace thanks to [Chengrui Wang](https://github.com/crywang).

Note:
- There is [a PR for OpenCV adding this model](https://github.com/opencv/opencv/pull/20422) to work with OpenCV DNN in C++ implementation.
- Support 5-landmark warp for now.
- `demo.py` requires [../face_detection_yunet](../face_detection_yunet) to run.

## Demo

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