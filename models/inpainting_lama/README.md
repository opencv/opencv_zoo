# Lama

LaMa is a very lightweight yet powerful image inpainting model.

Notes:

- Model source: [ONNX](https://huggingface.co/Carve/LaMa-ONNX/blob/main/lama_fp32.onnx).

## Requirements 
Install latest OpenCV >=5.0.0 and CMake >= 3.22.1 to get started with.

## Demo

### Python

Run the following command to try the demo:

```shell
# usage
python demo.py --input /path/to/image

# get help regarding various parameters
python demo.py --help
```

### C++

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# usage
./build/demo --input=/path/to/image
# get help messages
./build/demo -h
```

### Example outputs

![chicky](./example_outputs/squirrel_output.jpg)

## License

All files in this directory are licensed under [Apache License](./LICENSE).

## Reference

- https://github.com/advimman/lama