# License Plate Detection with YuNet

This model is contributed by Dong Xu (徐栋) from [watrix.ai](watrix.ai) (银河水滴).

Please note that the model is trained with Chinese license plates, so the detection results of other license plates with this model may be limited.

## Demo

Run the following command to try the demo:

```shell
# detect on camera input
python demo.py
# detect on an image
python demo.py --input /path/to/image -v
# get help regarding various parameters
python demo.py --help
```

### Example outputs

![lpd](./example_outputs/lpd_yunet_demo.gif)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE)

## Reference

- https://github.com/ShiqiYu/libfacedetection.train
