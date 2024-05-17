# image_segmentation_efficientsam

EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything

Notes:
- The current implementation of the EfficientSAM demo uses the EfficientSAM-Ti model, which is specifically tailored for scenarios requiring higher speed and lightweight. 


## Demo

### Python
Run the following command to try the demo:

```shell
python demo.py --input /path/to/image
```

Click only **once** on the object you wish to segment in the displayed image. After the click, the segmentation result will be shown in a new window.

## Result

Here are some of the sample results that were observed using the model:

![test1_res.jpg](./example_outputs/example1.png)
![test2_res.jpg](./example_outputs/example2.png)

## Model metrics:

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

#### Contributor Details

## Reference

- https://arxiv.org/abs/2312.00863
- https://github.com/yformer/EfficientSAM