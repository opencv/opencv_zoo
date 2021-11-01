# CRNN

An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

Note:
- Model source: https://docs.opencv.org/4.5.2/d9/d1e/tutorial_dnn_OCR.html.
- For details on training this model, please visit https://github.com/zihaomu/deep-text-recognition-benchmark, which can only recognize english words.

## Demo

***NOTE***: This demo uses [text_detection_db](../text_detection_db) as text detector.

Run the following command to try the demo:
```shell
# detect on camera input
python demo.py
# detect on an image
python demo.py --input /path/to/image
```

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://arxiv.org/abs/1507.05717
- https://github.com/bgshih/crnn
- https://github.com/meijieru/crnn.pytorch
- https://github.com/zihaomu/deep-text-recognition-benchmark
- https://docs.opencv.org/4.5.2/d9/d1e/tutorial_dnn_OCR.html