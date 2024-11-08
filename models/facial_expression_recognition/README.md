
# Progressive Teacher

Progressive Teacher: [Boosting Facial Expression Recognition by A Semi-Supervised Progressive Teacher](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=OCwcfAwAAAAJ&citation_for_view=OCwcfAwAAAAJ:u5HHmVD_uO8C)

Note:
- Progressive Teacher is contributed by [Jing Jiang](https://scholar.google.com/citations?user=OCwcfAwAAAAJ&hl=zh-CN).
-  [MobileFaceNet](https://link.springer.com/chapter/10.1007/978-3-319-97909-0_46) is used as the backbone and the model is able to classify seven basic facial expressions (angry, disgust, fearful, happy, neutral, sad, surprised).
- [facial_expression_recognition_mobilefacenet_2022july.onnx](https://github.com/opencv/opencv_zoo/raw/master/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx) is implemented thanks to [Chengrui Wang](https://github.com/crywang).
- `facial_expression_recognition_mobilefacenet_2022july_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.

Results of accuracy evaluation on [RAF-DB](http://whdeng.cn/RAF/model1.html).

| Models      | Accuracy | 
|-------------|----------|
| Progressive Teacher       | 88.27%  |


## Demo

***NOTE***: This demo uses [../face_detection_yunet](../face_detection_yunet) as face detector, which supports 5-landmark detection for now (2021sep).

### Python
Run the following command to try the demo:
```shell
# recognize the facial expression on images
python demo.py --input /path/to/image -v
```

### C++

Install latest OpenCV and CMake >= 3.24.0 to get started with:

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# detect on camera input
./build/opencv_zoo_face_expression_recognition
# detect on an image
./build/opencv_zoo_face_expression_recognition -i=/path/to/image
# get help messages
./build/opencv_zoo_face_expression_recognition -h
```

### Example outputs

Note: Zoom in to to see the recognized facial expression in the top-left corner of each face boxes.

![fer demo](./example_outputs/selfie.jpg)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://ieeexplore.ieee.org/abstract/document/9629313
