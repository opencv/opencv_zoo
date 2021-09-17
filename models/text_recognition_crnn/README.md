# CRNN

An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

`text_recognition_crnn.onnx` is trained using the code from https://github.com/zihaomu/deep-text-recognition-benchmark, which can only recognize english words. It is obtained from https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr and renamed from `CRNN_VGG_BiLSTM_CTC.onnx`. Visit https://docs.opencv.org/4.5.2/d9/d1e/tutorial_dnn_OCR.html for more information.

## Demo

***NOTE**: This demo use [text_detection_db](../text_detection_db) as text detector.

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