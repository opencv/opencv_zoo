# YOLOX

Nanodet: YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities. YOLOX is a high-performing object detector, an improvement to the existing YOLO series. YOLO series are in constant exploration of techniques to improve the object detection techniques for optimal speed and accuracy trade-off for real-time applications.

Key features of the YOLOX object detector
- **Anchor-free detectors** significantly reduce the number of design parameters
- **A decoupled head for classification, regression, and localization** improves the convergence speed
- **SimOTA advanced label assignment strategy** reduces training time and avoids additional solver hyperparameters
- **Strong data augmentations like MixUp and Mosiac** to boost YOLOX performance

**Note**:
- This version of YoloX: YoloX_s
- `object_detection_yolox_2022nov_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.


## Demo

### Python

Run the following command to try the demo: 
```shell
# detect on camera input
python demo.py
# detect on an image
python demo.py --input /path/to/image -v
```
Note: 
- image result saved as "result.jpg"
- this model requires `opencv-python>=4.8.0`

### C++

Install latest OpenCV and CMake >= 3.24.0 to get started with:

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# detect on camera input
./build/opencv_zoo_object_detection_yolox
# detect on an image
./build/opencv_zoo_object_detection_yolox -m=/path/to/model -i=/path/to/image -v
# get help messages
./build/opencv_zoo_object_detection_yolox -h
```


## Results

Here are some of the sample results that were observed using the model (**yolox_s.onnx**),

![1_res.jpg](./example_outputs/1_res.jpg)
![2_res.jpg](./example_outputs/2_res.jpg)
![3_res.jpg](./example_outputs/3_res.jpg)

Check [benchmark/download_data.py](../../benchmark/download_data.py) for the original images.

## Model metrics:

The model is evaluated on [COCO 2017 val](https://cocodataset.org/#download). Results are showed below:

<table>
<tr><th>Average Precision </th><th>Average Recall</th></tr>
<tr><td>

|  area  |  IoU  |  Average Precision(AP)  |
|:-------|:------|:------------------------|
|  all  |  0.50:0.95  |  0.405  |
|  all  |  0.50  |  0.593  |
|  all  |  0.75  |  0.437  |
|  small  |  0.50:0.95  |  0.232  |
|  medium  |  0.50:0.95  |  0.448  |
|  large  |  0.50:0.95  |  0.541  |

 </td><td>

|   area |  IoU  |  Average Recall(AR)  |
|:-------|:------|:----------------|
|  all  |  0.50:0.95  |  0.326  |
|  all  |  0.50:0.95  |  0.531  |
|  all  |  0.50:0.95 |  0.574  |
|  small  |  0.50:0.95  |  0.365  |
|  medium  |  0.50:0.95  |  0.634  |
|  large  |  0.50:0.95  |  0.724  |
</td></tr> </table>

| class         | AP     | class        | AP     | class          | AP     |
|:--------------|:-------|:-------------|:-------|:---------------|:-------|
| person        | 54.109 | bicycle      | 31.580 | car            | 40.447 |
| motorcycle    | 43.477 | airplane     | 66.070 | bus            | 64.183 |
| train         | 64.483 | truck        | 35.110 | boat           | 24.681 |
| traffic light | 25.068 | fire hydrant | 64.382 | stop sign      | 65.333 |
| parking meter | 48.439 | bench        | 22.653 | bird           | 33.324 |
| cat           | 66.394 | dog          | 60.096 | horse          | 58.080 |
| sheep         | 49.456 | cow          | 53.596 | elephant       | 65.574 |
| bear          | 70.541 | zebra        | 66.461 | giraffe        | 66.780 |
| backpack      | 13.095 | umbrella     | 41.614 | handbag        | 12.865 |
| tie           | 29.453 | suitcase     | 39.089 | frisbee        | 61.712 |
| skis          | 21.623 | snowboard    | 31.326 | sports ball    | 39.820 |
| kite          | 41.410 | baseball bat | 27.311 | baseball glove | 36.661 |
| skateboard    | 49.374 | surfboard    | 35.524 | tennis racket  | 45.569 |
| bottle        | 37.270 | wine glass   | 33.088 | cup            | 39.835 |
| fork          | 31.620 | knife        | 15.265 | spoon          | 14.918 |
| bowl          | 43.251 | banana       | 27.904 | apple          | 17.630 |
| sandwich      | 32.789 | orange       | 29.388 | broccoli       | 23.187 |
| carrot        | 23.114 | hot dog      | 33.716 | pizza          | 52.541 |
| donut         | 47.980 | cake         | 36.160 | chair          | 29.707 |
| couch         | 46.175 | potted plant | 24.781 | bed            | 44.323 |
| dining table  | 30.022 | toilet       | 64.237 | tv             | 57.301 |
| laptop        | 58.362 | mouse        | 57.774 | remote         | 24.271 |
| keyboard      | 48.020 | cell phone   | 32.376 | microwave      | 57.220 |
| oven          | 36.168 | toaster      | 28.735 | sink           | 38.159 |
| refrigerator  | 52.876 | book         | 15.030 | clock          | 48.622 |
| vase          | 37.013 | scissors     | 26.307 | teddy bear     | 45.676 |
| hair drier    | 7.255  | toothbrush   | 19.374 |                |        |

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

#### Contributor Details

- Google Summer of Code'22
- Contributor: Sri Siddarth Chakaravarthy
- Github Profile: https://github.com/Sidd1609
- Organisation: OpenCV
- Project: Lightweight object detection models using OpenCV 

## Reference

- YOLOX article: https://arxiv.org/abs/2107.08430
- YOLOX weight and scripts for training: https://github.com/Megvii-BaseDetection/YOLOX
- YOLOX blog: https://arshren.medium.com/yolox-new-improved-yolo-d430c0e4cf20
- YOLOX-lite: https://github.com/TexasInstruments/edgeai-yolox
