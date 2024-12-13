# WeChatQRCode

WeChatQRCode for detecting and parsing QR Code, contributed by [WeChat Computer Vision Team (WeChatCV)](https://github.com/WeChatCV). Visit [opencv/opencv_contrib/modules/wechat_qrcode](https://github.com/opencv/opencv_contrib/tree/master/modules/wechat_qrcode) for more details.

Notes:

- Model source: [opencv/opencv_3rdparty:wechat_qrcode_20210119](https://github.com/opencv/opencv_3rdparty/tree/wechat_qrcode_20210119)
- The APIs `cv::wechat_qrcode::WeChatQRCode` (C++) & `cv.wechat_qrcode_WeChatQRCode` (Python) are both designed to run on default backend (OpenCV) and target (CPU) only. Therefore, benchmark results of this model are only available on CPU devices, until the APIs are updated with setting backends and targets.

## Demo

### Python

Run the following command to try the demo:

```shell
# detect on camera input
python demo.py
# detect on an image
python demo.py --input /path/to/image -v

# get help regarding various parameters
python demo.py --help
```

### C++

Install latest OpenCV (with opencv_contrib) and CMake >= 3.24.0 to get started with:

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# detect on camera input
./build/demo
# detect on an image
./build/demo -i=/path/to/image -v
# get help messages
./build/demo -h
```

### Example outputs

![webcam demo](./example_outputs/wechat_qrcode_demo.gif)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference:

- https://github.com/opencv/opencv_contrib/tree/master/modules/wechat_qrcode
- https://github.com/opencv/opencv_3rdparty/tree/wechat_qrcode_20210119
