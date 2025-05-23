# DexiNed

DexiNed is a Convolutional Neural Network (CNN) architecture for edge detection.

Notes:

- Model source: [ONNX](https://drive.google.com/file/d/1u_qXqXqaIP_SqdGaq4CbZyjzkZb02XTs/view).
- Model source: [.pth](https://drive.google.com/file/d/1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu/view).
- This ONNX model has fixed input shape, but OpenCV DNN infers on the exact shape of input image. See https://github.com/opencv/opencv_zoo/issues/44 for more information.

## Requirements 
Install latest OpenCV >=5.0.0 and CMake >= 3.22.2 to get started with.

## Demo

### Python

Run the following command to try the demo:

```shell
# detect on camera input
python demo.py
# detect on an image
python demo.py --input /path/to/image

# get help regarding various parameters
python demo.py --help
```

### C++

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# detect on camera input
./build/demo
# detect on an image
./build/demo --input=/path/to/image
# get help messages
./build/demo -h
```

### Example outputs

![chicky](./example_outputs/chicky_output.jpg)

## License

All files in this directory are licensed under [MIT License](./LICENSE).

## Reference

- https://github.com/xavysp/DexiNed