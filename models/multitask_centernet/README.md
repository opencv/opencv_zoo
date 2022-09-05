# MCN

Multitask-Centernet (MCN)  is a multi-task network (MTN). Studies have shown that training with multiple tasks linked to each other can sometimes even improve the quality of training and prediction compared to single-task learning (STL). When the network receives the same type of input, it is likely to extract similar features. In this case, a shared backbone can take advantage of the similar semantics of these input features.

Notes:
- Model source: [here](https://drive.google.com/file/d/1HmYZ_HccS41kolqW9KHfcKEQKjXSBZnY/view?usp=sharing).
- For details on training this model, please visit my home page

## Demo

Run the following command to try the demo:
```shell
# detect on an image
python demo.py --input /path/to/image
```

### Example outputs

![detection and pose estimation demo](./examples/ori_vis_0.png)

![semantic segmentation demo](./examples/ori_vis_masks_0.png)

## License

All files in this directory are licensed under [MIT License](./LICENSE).

## Reference

- https://arxiv.org/abs/2108.05060v2

