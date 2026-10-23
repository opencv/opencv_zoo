# License Plate Detection with YuNet

This model is contributed by Dong Xu (徐栋) from [watrix.ai](watrix.ai) (银河水滴).

Please note that the model is trained with Chinese license plates, so the detection results of other license plates with this model may be limited.

**Note**:
- `license_plate_detection_lpd_yunet_2023mar_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.

Results of accuracy evaluation with [tools/eval](../../tools/eval).

Protocol: CCPD official detection metric (Xu et al., ECCV 2018). One box per image is correct iff IoU > 0.7. `Overall` is precision on the union of the six official test subsets (DB, Blur, FN, Rotate, Tilt, Challenge). LPD-YuNet does not publish a claimed accuracy; numbers below are measured with OpenCV DNN.

| Models          | Overall | DB  | Blur | FN  | Rotate | Tilt | Challenge |
| --------------- | ------- | --- | ---- | --- | ------ | ---- | --------- |
| LPD-YuNet       | TBD     | TBD | TBD  | TBD | TBD    | TBD  | TBD       |
| LPD-YuNet block | TBD     | TBD | TBD  | TBD | TBD    | TBD  | TBD       |
| LPD-YuNet quant | TBD     | TBD | TBD  | TBD | TBD    | TBD  | TBD       |

*: 'quant' stands for 'quantized'.
**: 'block' stands for 'blockwise quantized'.

## Demo

Run the following command to try the demo:

```shell
# detect on camera input
python demo.py
# detect on an image
python demo.py --input /path/to/image -v
# get help regarding various parameters
python demo.py --help
### Example outputs

![lpd](./example_outputs/lpd_yunet_demo.gif)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE)

## Reference

- https://github.com/ShiqiYu/libfacedetection.train

### Example outputs

![lpd](./example_outputs/lpd_yunet_demo.gif)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE)

## Reference

- https://github.com/ShiqiYu/libfacedetection.train
