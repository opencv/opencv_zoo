

# MAXIM: Multi-Axis MLP for Image Processing (CVPR 2022 Oral)

fixed model input to 512*640, padding with  'reflect' mode

Notes:
- 
- This ONNX model has fixed input shape (512*640), any image smaller than this will be padded with reflection mode, and any image larger than this will cause error.

## Demo

Run the following command to try the demo:
```shell
# detect on sample image 1
python demo.py
# detect on an image
python demo.py --input /path/to/image
```

### Example outputs

![sample image 1](./examples/input/image1.png)
![sample dehazed image 1](./examples/output/dehazed_image1.png)
![sample image 2](./examples/input/image2.png)
![sample dehazed image 2](./examples/output/dehazed_image2.png)


## License

All files in this directory are licensed under  [Apache 2.0 License](./LICENSE).

## Reference

https://github.com/google-research/maxim