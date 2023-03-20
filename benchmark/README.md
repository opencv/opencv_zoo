# OpenCV Zoo Benchmark

Benchmarking the speed of OpenCV DNN inferring different models in the zoo. Result of each model includes the time of its preprocessing, inference and postprocessing stages.

Data for benchmarking will be downloaded and loaded in [data](./data) based on given config.

## Preparation

1. Install `python >= 3.6`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download data for benchmarking.
    1. Download all data: `python download_data.py`
    2. Download one or more specified data: `python download_data.py face text`. Available names can be found in `download_data.py`.
    3. You can also download all data from https://pan.baidu.com/s/18sV8D4vXUb2xC9EG45k7bg (code: pvrw). Please place and extract data packages under [./data](./data).

## Benchmarking

**Linux**:

```shell
export PYTHONPATH=$PYTHONPATH:.. 
python benchmark.py --cfg ./config/face_detection_yunet.yaml
```

**Windows**:
- CMD
    ```shell
    set PYTHONPATH=%PYTHONPATH%;..
    python benchmark.py --cfg ./config/face_detection_yunet.yaml
    ```

- PowerShell
    ```shell
    $env:PYTHONPATH=$env:PYTHONPATH+";.."
    python benchmark.py --cfg ./config/face_detection_yunet.yaml
    ```
<!--
Omit `--cfg` if you want to benchmark all included models:
```shell
PYTHONPATH=.. python benchmark.py
```
-->

## More Results

Benchmark is done with latest `opencv-python==4.7.0.72` and `opencv-contrib-python==4.7.0.72` on the following platforms. Some models are excluded because of support issues.

### Rasberry Pi 4B (CPU)

```
$ python benchmark.py --all --model_exclude license_plate_detection_lpd_yunet_2023mar-act_int8-wt_int8-quantized.onnx:human_segmentation_pphumanseg_2023mar-act_int8-wt_int8-quantized.onnx
Benchmarking ...
mean=5.43	median=5.41	min=5.38	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar.onnx']
mean=6.09	median=6.19	min=5.38	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar-act_int8-wt_int8-quantized.onnx']
mean=78.83	median=78.83	min=78.43	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec.onnx']
mean=92.56	median=94.20	min=78.43	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec-act_int8-wt_int8-quantized.onnx']
mean=32.53	median=32.51	min=32.33	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
mean=38.61	median=39.19	min=32.33	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july-int8-quantized.onnx']
mean=43.57	median=43.59	min=43.26	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
mean=46.69	median=47.12	min=43.26	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb-act_int8-wt_int8-quantized.onnx']
mean=73.13	median=73.79	min=72.71	input size=[192, 192]	model: PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
mean=72.33	median=72.49	min=72.13	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
mean=66.56	median=66.92	min=61.25	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
mean=67.47	median=62.02	min=61.25	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr-int8-quantized.onnx']
mean=66.01	median=65.53	min=61.25	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr-int8-quantized.onnx']
mean=463.93	median=463.23	min=459.88	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
mean=441.38	median=434.24	min=373.27	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan-act_int8-wt_int8-quantized.onnx']
mean=167.70	median=173.37	min=143.13	input size=[320, 240]	model: LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
mean=225.10	median=223.10	min=208.92	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov.onnx']
mean=226.14	median=226.32	min=206.70	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
mean=1805.87	median=1827.43	min=1717.68	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov.onnx']
mean=1737.66	median=1789.99	min=1575.72	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov_int8.onnx']
mean=712.94	median=701.07	min=580.18	input size=[1280, 720]	model: DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
mean=86.83	median=76.78	min=65.25	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
mean=107.26	median=98.07	min=65.25	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb-act_int8-wt_int8-quantized.onnx']
mean=625.56	median=622.26	min=599.92	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov.onnx']
mean=512.76	median=519.71	min=353.70	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov-act_int8-wt_int8-quantized.onnx']
mean=5.71	median=5.76	min=5.63	input size=[100, 100]	model: WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
mean=1862.75	median=1849.26	min=1812.02	input size=[640, 480]	model: DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
mean=1878.45	median=1880.35	min=1812.02	input size=[640, 480]	model: DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
mean=271.85	median=276.14	min=259.73	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
mean=297.48	median=300.89	min=259.73	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
mean=278.11	median=357.83	min=229.71	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@1933.986] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
mean=272.00	median=289.96	min=229.71	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
mean=261.77	median=273.49	min=228.71	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
mean=254.22	median=253.40	min=226.83	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
mean=251.15	median=260.79	min=226.83	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov-act_int8-wt_int8-quantized.onnx']
mean=242.05	median=245.09	min=190.53	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```
