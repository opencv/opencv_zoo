# Nanodet

Nanodet: NanoDet is a FCOS-style one-stage anchor-free object detection model which using Generalized Focal Loss as classification and regression loss.In NanoDet-Plus, we propose a novel label assignment strategy with a simple assign guidance module (AGM) and a dynamic soft label assigner (DSLA) to solve the optimal label assignment problem in lightweight model training.

**Note**:
- This version of nanodet: Nanodet-m-plus-1.5x_416
- `object_detection_nanodet_2022nov_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.


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

### C++

Install latest OpenCV and CMake >= 3.24.0 to get started with:

```shell
# A typical and default installation path of OpenCV is /usr/local
cmake -B build -D OPENCV_INSTALLATION_PATH=/path/to/opencv/installation .
cmake --build build

# detect on camera input
./build/opencv_zoo_object_detection_nanodet
# detect on an image
./build/opencv_zoo_object_detection_nanodet -i=/path/to/image
# get help messages
./build/opencv_zoo_object_detection_nanodet -h
```


## Results

Here are some of the sample results that were observed using the model,

![test1_res.jpg](./example_outputs/1_res.jpg)
![test2_res.jpg](./example_outputs/2_res.jpg)

Check [benchmark/download_data.py](../../benchmark/download_data.py) for the original images.

Video inference result,
![WebCamR.gif](./example_outputs/WebCamR.gif)

## Model metrics:

The model is evaluated on [COCO 2017 val](https://cocodataset.org/#download). Results are showed below:

<table>
<tr><th>Average Precision </th><th>Average Recall</th></tr>
<tr><td>
  
|  area  |  IoU  |  Average Precision(AP)  |
|:-------|:------|:------------------------|
|  all  |  0.50:0.95  |  0.304  |
|  all  |  0.50  |  0.459  |
|  all  |  0.75  |  0.317  |
|  small  |  0.50:0.95  |  0.107  |
|  medium  |  0.50:0.95  |  0.322  |
|  large  |  0.50:0.95  |  0.478  |
 
 </td><td>

  area  |  IoU  |  Average Recall  |
|:-------|:------|:----------------|
|  all  |  0.50:0.95  |  0.278  |
|  all  |  0.50:0.95  |  0.434  |
|  all  |  0.50:0.95 |  0.462  |
|  small  |  0.50:0.95  |  0.198  |
|  medium  |  0.50:0.95  |  0.510  |
|  large  |  0.50:0.95  |  0.702  |
</td></tr> </table>

| class         | AP50   | mAP   | class          | AP50   | mAP   |
|:--------------|:-------|:------|:---------------|:-------|:------|
| person        | 67.5   | 41.8  | bicycle        | 35.4   | 18.8  |
| car           | 45.0   | 25.4  | motorcycle     | 58.9   | 33.1  |
| airplane      | 77.3   | 58.9  | bus            | 68.8   | 56.4  |
| train         | 81.1   | 60.5  | truck          | 38.6   | 24.7  |
| boat          | 35.5   | 16.7  | traffic light  | 30.5   | 14.0  |
| fire hydrant  | 69.8   | 54.5  | stop sign      | 60.9   | 54.6  |
| parking meter | 55.1   | 38.5  | bench          | 26.8   | 15.9  |
| bird          | 38.3   | 23.6  | cat            | 82.5   | 62.1  |
| dog           | 67.0   | 51.4  | horse          | 64.3   | 44.2  |
| sheep         | 57.7   | 35.8  | cow            | 61.2   | 39.9  |
| elephant      | 79.9   | 56.2  | bear           | 81.8   | 63.0  |
| zebra         | 85.4   | 59.5  | giraffe        | 84.1   | 59.9  |
| backpack      | 12.4   | 5.9   | umbrella       | 46.5   | 28.8  |
| handbag       | 8.4    | 3.7   | tie            | 35.2   | 19.6  |
| suitcase      | 38.1   | 23.8  | frisbee        | 60.7   | 43.9  |
| skis          | 30.5   | 14.5  | snowboard      | 32.3   | 18.2  |
| sports ball   | 37.6   | 24.5  | kite           | 51.1   | 30.4  |
| baseball bat  | 28.9   | 13.6  | baseball glove | 40.1   | 21.6  |
| skateboard    | 59.4   | 35.2  | surfboard      | 47.9   | 26.6  |
| tennis racket | 55.2   | 30.5  | bottle         | 34.7   | 20.2  |
| wine glass    | 27.8   | 16.3  | cup            | 35.5   | 23.7  |
| fork          | 25.9   | 14.8  | knife          | 10.9   | 5.6   |
| spoon         | 8.7    | 4.1   | bowl           | 42.8   | 29.4  |
| banana        | 35.5   | 18.5  | apple          | 19.4   | 12.9  |
| sandwich      | 46.7   | 33.4  | orange         | 35.2   | 25.9  |
| broccoli      | 36.4   | 19.1  | carrot         | 30.9   | 17.8  |
| hot dog       | 42.7   | 29.3  | pizza          | 61.0   | 44.9  |
| donut         | 47.3   | 34.0  | cake           | 39.9   | 24.4  |
| chair         | 28.8   | 16.1  | couch          | 60.5   | 42.6  |
| potted plant  | 29.0   | 15.3  | bed            | 63.3   | 46.0  |
| dining table  | 39.6   | 27.5  | toilet         | 71.3   | 55.3  |
| tv            | 66.5   | 48.1  | laptop         | 62.6   | 46.9  |
| mouse         | 63.5   | 44.1  | remote         | 19.8   | 10.3  |
| keyboard      | 62.1   | 41.5  | cell phone     | 33.7   | 22.8  |
| microwave     | 54.9   | 39.6  | oven           | 48.1   | 30.4  |
| toaster       | 30.0   | 16.4  | sink           | 44.5   | 27.8  |
| refrigerator  | 63.2   | 46.1  | book           | 18.4   | 7.3   |
| clock         | 57.8   | 35.8  | vase           | 33.7   | 22.1  |
| scissors      | 27.8   | 17.8  | teddy bear     | 54.1   | 35.4  |
| hair drier    | 2.9    | 1.1   | toothbrush     | 13.1   | 8.2   |

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

#### Contributor Details

- Google Summer of Code'22
- Contributor: Sri Siddarth Chakaravarthy
- Github Profile: https://github.com/Sidd1609
- Organisation: OpenCV
- Project: Lightweight object detection models using OpenCV 

## Reference

- Nanodet: https://zhuanlan.zhihu.com/p/306530300
- Nanodet Plus: https://zhuanlan.zhihu.com/p/449912627
- Nanodet weight and scripts for training: https://github.com/RangiLyu/nanodet
