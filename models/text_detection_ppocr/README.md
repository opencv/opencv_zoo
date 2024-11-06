# PP-OCRv3 Text Detection

PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System.

**Note**:

- The int8 quantization model may produce unstable results due to some loss of accuracy.
- Original Paddle Models source of English: [here](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar).
- Original Paddle Models source of Chinese: [here](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar).
- `IC15` in the filename means the model is trained on [IC15 dataset](https://rrc.cvc.uab.es/?ch=4&com=introduction), which can detect English text instances only.
- `TD500` in the filename means the model is trained on [TD500 dataset](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)), which can detect both English & Chinese instances.
- Visit https://docs.opencv.org/master/d4/d43/tutorial_dnn_text_spotting.html for more information.
- `text_detection_xx_ppocrv3_2023may_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.

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

Install latest OpenCV and CMake >= 3.24.0 to get started with:

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build
# detect on camera input
./build/opencv_zoo_text_detection_ppocr -m=/path/to/model
# detect on an image
./build/opencv_zoo_text_detection_ppocr -m=/path/to/model -i=/path/to/image -v
# get help messages
./build/opencv_zoo_text_detection_ppocr -h
```

### Example outputs

![mask](./example_outputs/mask.jpg)

![gsoc](./example_outputs/gsoc.jpg)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://arxiv.org/abs/2206.03001
- https://github.com/PaddlePaddle/PaddleOCR
- https://docs.opencv.org/master/d4/d43/tutorial_dnn_text_spotting.html
