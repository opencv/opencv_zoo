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

### Intel 12700K

CPU:

```
$ python benchmark.py --all --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean=0.72	median=0.70	min=0.62	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar.onnx']
mean=0.90	median=0.90	min=0.62	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar_int8.onnx']
mean=6.04	median=6.08	min=5.59	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec.onnx']
mean=7.34	median=7.94	min=5.59	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec_int8.onnx']
mean=3.16	median=3.37	min=2.87	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
mean=4.14	median=3.98	min=2.87	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
mean=4.68	median=4.59	min=4.13	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
mean=4.90	median=4.71	min=4.13	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
mean=8.81	median=7.37	min=6.53	input size=[192, 192]	model: PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
mean=5.90	median=5.87	min=5.10	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
mean=5.97	median=6.66	min=5.10	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
mean=6.34	median=6.01	min=5.10	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
mean=6.52	median=7.02	min=5.10	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
mean=34.71	median=35.41	min=32.15	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
mean=35.03	median=35.31	min=32.15	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
mean=8.63	median=7.99	min=7.54	input size=[320, 240]	model: LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
mean=66.03	median=64.57	min=61.72	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov.onnx']
mean=65.78	median=68.48	min=61.72	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
mean=141.20	median=151.61	min=118.97	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov.onnx']
mean=141.85	median=151.42	min=118.97	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov_int8.onnx']
mean=29.05	median=28.50	min=22.84	input size=[1280, 720]	model: DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
mean=6.29	median=6.27	min=5.90	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
mean=8.65	median=9.08	min=5.90	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
mean=30.39	median=29.95	min=29.43	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov.onnx']
mean=30.71	median=30.73	min=29.43	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
mean=1.29	median=1.23	min=1.12	input size=[100, 100]	model: WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
mean=71.03	median=70.79	min=64.87	input size=[640, 480]	model: DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
mean=72.31	median=76.96	min=64.87	input size=[640, 480]	model: DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
mean=21.33	median=25.27	min=16.34	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
mean=23.07	median=22.38	min=16.34	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
mean=20.16	median=26.50	min=11.49	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@144.899] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
mean=20.58	median=24.46	min=11.49	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
mean=19.28	median=16.05	min=11.49	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
mean=19.27	median=16.80	min=11.49	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
mean=19.55	median=17.81	min=11.49	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
mean=18.94	median=21.04	min=11.09	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Rasberry Pi 4B

CPU:

```
$ python benchmark.py --all --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean=5.43	median=5.41	min=5.38	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar.onnx']
mean=6.09	median=6.19	min=5.38	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar_int8.onnx']
mean=78.83	median=78.83	min=78.43	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec.onnx']
mean=92.56	median=94.20	min=78.43	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec_int8.onnx']
mean=32.53	median=32.51	min=32.33	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
mean=38.61	median=39.19	min=32.33	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
mean=43.57	median=43.59	min=43.26	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
mean=46.69	median=47.12	min=43.26	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
mean=73.13	median=73.79	min=72.71	input size=[192, 192]	model: PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
mean=72.33	median=72.49	min=72.13	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
mean=66.56	median=66.92	min=61.25	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
mean=67.47	median=62.02	min=61.25	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
mean=66.01	median=65.53	min=61.25	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
mean=463.93	median=463.23	min=459.88	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
mean=441.38	median=434.24	min=373.27	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
mean=167.70	median=173.37	min=143.13	input size=[320, 240]	model: LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
mean=225.10	median=223.10	min=208.92	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov.onnx']
mean=226.14	median=226.32	min=206.70	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
mean=1805.87	median=1827.43	min=1717.68	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov.onnx']
mean=1737.66	median=1789.99	min=1575.72	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov_int8.onnx']
mean=712.94	median=701.07	min=580.18	input size=[1280, 720]	model: DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
mean=86.83	median=76.78	min=65.25	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
mean=107.26	median=98.07	min=65.25	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
mean=625.56	median=622.26	min=599.92	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov.onnx']
mean=512.76	median=519.71	min=353.70	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
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
mean=251.15	median=260.79	min=226.83	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
mean=242.05	median=245.09	min=190.53	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Jetson Nano B01

CPU:

```
$ python3 benchmark.py --all --cfg_exclude wechat --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean=5.29	median=5.29	min=5.23	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar.onnx']
mean=6.03	median=6.18	min=5.23	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar_int8.onnx']
mean=65.42	median=65.37	min=65.20	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec.onnx']
mean=79.12	median=79.78	min=65.20	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec_int8.onnx']
mean=28.28	median=28.30	min=28.12	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
mean=34.87	median=35.71	min=28.12	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
mean=40.87	median=40.84	min=40.71	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
mean=44.14	median=44.28	min=40.71	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
mean=65.94	median=65.86	min=65.72	input size=[192, 192]	model: PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
mean=69.17	median=69.04	min=68.84	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
mean=62.20	median=62.36	min=55.52	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
mean=66.22	median=55.93	min=55.52	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
mean=65.34	median=75.35	min=55.52	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
mean=374.52	median=383.78	min=367.81	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
mean=389.18	median=408.69	min=367.81	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
mean=130.63	median=131.11	min=128.36	input size=[320, 240]	model: LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
mean=214.30	median=212.83	min=212.53	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov.onnx']
mean=216.49	median=220.40	min=212.53	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
mean=1225.13	median=1234.78	min=1213.85	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov.onnx']
mean=1256.33	median=1261.81	min=1213.85	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov_int8.onnx']
mean=471.54	median=492.74	min=450.43	input size=[1280, 720]	model: DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
mean=70.52	median=70.05	min=69.65	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
mean=82.59	median=82.46	min=69.65	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
mean=420.32	median=414.19	min=410.95	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov.onnx']
mean=375.09	median=384.11	min=333.25	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
mean=1106.86	median=1093.50	min=1090.64	input size=[640, 480]	model: DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
mean=1104.27	median=1106.12	min=1086.68	input size=[640, 480]	model: DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
mean=281.11	median=296.92	min=276.23	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
mean=297.92	median=302.00	min=276.23	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
mean=277.85	median=313.29	min=240.38	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@1524.821] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
mean=275.41	median=299.86	min=240.38	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
mean=265.87	median=241.65	min=230.57	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
mean=264.09	median=255.31	min=230.57	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
mean=264.84	median=276.17	min=230.57	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
mean=259.24	median=249.23	min=224.99	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

GPU (CUDA-FP32):
```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 1
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA
mean=11.88	median=11.42	min=9.62	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar.onnx']
mean=24.64	median=24.45	min=24.25	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec.onnx']
mean=20.56	median=21.16	min=18.78	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
mean=41.09	median=46.72	min=40.55	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
mean=89.74	median=86.81	min=84.49	input size=[192, 192]	model: PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
mean=69.14	median=80.06	min=68.81	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
mean=62.03	median=62.19	min=55.37	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
mean=146.98	median=148.92	min=144.63	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
mean=53.37	median=54.02	min=51.89	input size=[320, 240]	model: LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
mean=213.78	median=214.45	min=212.26	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov.onnx']
mean=1233.68	median=1244.26	min=1223.56	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov.onnx']
mean=76.43	median=75.60	min=74.54	input size=[1280, 720]	model: DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
mean=61.60	median=65.97	min=58.57	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
mean=127.71	median=127.58	min=125.22	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov.onnx']
mean=303.00	median=303.69	min=297.90	input size=[640, 480]	model: DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
mean=301.84	median=299.22	min=297.08	input size=[640, 480]	model: DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
mean=57.34	median=53.10	min=51.87	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
mean=58.70	median=63.59	min=51.87	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
mean=45.11	median=57.01	min=21.63	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

GPU (CUDA-FP16):

```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 2
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA_FP16
mean=26.34	median=25.66	min=25.55	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar.onnx']
mean=111.40	median=111.36	min=110.84	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec.onnx']
mean=115.56	median=113.08	min=112.66	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
mean=40.77	median=40.71	min=40.55	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
mean=98.94	median=97.82	min=91.00	input size=[192, 192]	model: PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
mean=68.83	median=68.54	min=68.40	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
mean=61.39	median=61.97	min=55.02	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
mean=136.91	median=129.33	min=128.82	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
mean=363.36	median=364.91	min=358.08	input size=[320, 240]	model: LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
mean=215.61	median=225.95	min=212.47	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov.onnx']
mean=1221.35	median=1230.06	min=1209.94	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov.onnx']
mean=52.16	median=52.17	min=50.31	input size=[1280, 720]	model: DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
mean=213.30	median=218.13	min=208.08	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
mean=92.16	median=89.69	min=87.86	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov.onnx']
mean=345.59	median=345.03	min=340.60	input size=[640, 480]	model: DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
mean=345.38	median=352.14	min=337.76	input size=[640, 480]	model: DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
mean=49.97	median=49.72	min=45.73	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
mean=50.40	median=51.25	min=45.73	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
mean=39.68	median=55.06	min=20.87	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

### Khadas VIM3

CPU:

```
$ python3 benchmark.py --all --cfg_exclude wechat --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean=4.88	median=4.95	min=4.78	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar.onnx']
mean=5.28	median=5.36	min=4.78	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar_int8.onnx']
mean=61.12	median=61.16	min=58.08	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec.onnx']
mean=70.74	median=70.32	min=58.08	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec_int8.onnx']
mean=29.07	median=37.91	min=28.24	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
mean=34.33	median=33.97	min=28.24	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
mean=38.71	median=38.17	min=37.07	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
mean=41.70	median=42.95	min=37.07	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
mean=70.71	median=70.68	min=66.49	input size=[192, 192]	model: PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
mean=67.46	median=65.78	min=63.07	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
mean=58.79	median=57.54	min=50.30	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
mean=59.67	median=53.69	min=50.30	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
mean=57.67	median=57.02	min=50.30	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
mean=386.43	median=387.23	min=349.69	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
mean=355.17	median=343.12	min=327.48	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
mean=133.92	median=120.20	min=118.81	input size=[320, 240]	model: LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
mean=212.42	median=214.09	min=191.86	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov.onnx']
mean=212.72	median=219.80	min=191.86	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
mean=1303.31	median=1358.03	min=1140.52	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov.onnx']
mean=1188.27	median=1139.30	min=1085.11	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov_int8.onnx']
mean=450.78	median=513.62	min=389.86	input size=[1280, 720]	model: DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
mean=66.26	median=65.25	min=64.69	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
mean=79.20	median=80.33	min=64.69	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
mean=459.81	median=447.06	min=406.86	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov.onnx']
mean=369.01	median=430.87	min=299.09	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
mean=983.62	median=989.13	min=951.54	input size=[640, 480]	model: DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
mean=996.42	median=979.01	min=942.02	input size=[640, 480]	model: DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
mean=202.69	median=202.09	min=201.07	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
mean=217.55	median=219.96	min=201.07	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
mean=200.11	median=234.19	min=169.53	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@1307.031] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
mean=200.22	median=203.99	min=169.53	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
mean=193.37	median=173.03	min=169.53	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
mean=189.89	median=186.60	min=169.53	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
mean=189.19	median=204.07	min=169.53	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
mean=183.55	median=188.47	min=149.32	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

NPU:

```
python3 benchmark.py --all --int8 --cfg_overwrite_backend_target 3 --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_TIMVX
target=cv.dnn.DNN_TARGET_NPU
mean=5.71	median=5.70	min=5.63	input size=[160, 120]	model: YuNet with ['face_detection_yunet_2022mar_int8.onnx']
mean=77.06	median=77.23	min=76.26	input size=[150, 150]	model: SFace with ['face_recognition_sface_2021dec_int8.onnx']
mean=40.81	median=39.83	min=38.90	input size=[112, 112]	model: FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
mean=45.59	median=46.33	min=43.45	input size=[224, 224]	model: MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
mean=59.84	median=60.37	min=55.79	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
mean=56.32	median=57.64	min=51.82	input size=[224, 224]	model: MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
mean=337.28	median=337.06	min=327.04	input size=[224, 224]	model: PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
mean=193.64	median=208.00	min=161.97	input size=[416, 416]	model: NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
mean=1106.77	median=1116.50	min=1065.63	input size=[640, 640]	model: YoloX with ['object_detection_yolox_2022nov_int8.onnx']
mean=94.99	median=94.49	min=91.96	input size=[192, 192]	model: MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
mean=304.75	median=299.43	min=297.99	input size=[128, 256]	model: YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
mean=178.31	median=178.62	min=177.75	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
mean=183.10	median=184.61	min=177.75	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
mean=171.68	median=191.14	min=149.26	input size=[1280, 720]	model: CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```
