# DaSiamRPN

[Distractor-aware Siamese Networks for Visual Object Tracking](https://arxiv.org/abs/1808.06048)

Note:
- Model source: [opencv/samples/dnn/diasiamrpn_tracker.cpp](https://github.com/opencv/opencv/blob/ceb94d52a104c0c1287a43dfa6ba72705fb78ac1/samples/dnn/dasiamrpn_tracker.cpp#L5-L7)
- Visit https://github.com/foolwood/DaSiamRPN for training details.

## Demo

Run the following command to try the demo:
```shell
# track on camera input
python demo.py
# track on video input
python demo.py --input /path/to/video
```

### Example outputs

![webcam demo](./examples/dasiamrpn_demo.gif)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference:

- DaSiamRPN Official Repository: https://github.com/foolwood/DaSiamRPN
- Paper: https://arxiv.org/abs/1808.06048
- OpenCV API `TrackerDaSiamRPN` Doc: https://docs.opencv.org/4.x/de/d93/classcv_1_1TrackerDaSiamRPN.html
- OpenCV Sample: https://github.com/opencv/opencv/blob/4.x/samples/dnn/dasiamrpn_tracker.cpp