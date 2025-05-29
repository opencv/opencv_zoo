# image_segmentation_efficientsam

EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything

Notes:
- The current implementation of the EfficientSAM demo uses the EfficientSAM-Ti model, which is specifically tailored for scenarios requiring higher speed and lightweight.
- image_segmentation_efficientsam_ti_2024may.onnx(supports only single point infering)
  - MD5 value: 117d6a6cac60039a20b399cc133c2a60
  - SHA-256 value: e3957d2cd1422855f350aa7b044f47f5b3eafada64b5904ed330b696229e2943
- image_segmentation_efficientsam_ti_2025april.onnx
  - MD5 value: f23cecbb344547c960c933ff454536a3
  - SHA-256 value: 4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77
- image_segmentation_efficientsam_ti_2025april_int8.onnx
  - MD5 value: a1164f44b0495b82e9807c7256e95a50
  - SHA-256 value: 5ecc8d59a2802c32246e68553e1cf8ce74cf74ba707b84f206eb9181ff774b4e


## Demo

### Python
Run the following command to try the demo:

```shell
python demo.py --input /path/to/image
```

**Click** to select foreground points, **drag** to use box to select and **long press** to select background points on the object you wish to segment in the displayed image. After clicking the **Enter**, the segmentation result will be shown in a new window. Clicking the **Backspace** to clear all the prompts.

## Result

Here are some of the sample results that were observed using the model:

![test1_res.jpg](./example_outputs/example1.png)
![test2_res.jpg](./example_outputs/example2.png)

Video inference result:

![sam_present.gif](./example_outputs/sam_present.gif)

## Model metrics:

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

#### Contributor Details

## Reference

- https://arxiv.org/abs/2312.00863
- https://github.com/yformer/EfficientSAM
- https://github.com/facebookresearch/segment-anything