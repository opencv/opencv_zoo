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

# Single config
python benchmark.py --cfg ./config/face_detection_yunet.yaml

# All configs
python benchmark.py --all

# All configs but only fp32 models (--fp32, --fp16, --int8 are available for now)
python benchmark.py --all --fp32

# All configs but exclude some of them (fill with config name keywords, not sensitive to upper/lower case, seperate with colons)
python benchmark.py --all --cfg_exclude wechat
python benchmark.py --all --cfg_exclude wechat:crnn

# All configs but exclude some of the models (fill with exact model names, sensitive to upper/lower case, seperate with colons)
python benchmark.py --all --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx

# All configs with overwritten backend and target (run with --help to get available combinations)
python benchmark.py --all --cfg_overwrite_backend_target 1
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

## Detailed Results

Benchmark is done with latest `opencv-python==4.8.0.74` and `opencv-contrib-python==4.8.0.74` on the following platforms. Some models are excluded because of support issues.

### Intel 12700K

Specs: [details](https://www.intel.com/content/www/us/en/products/sku/134594/intel-core-i712700k-processor-25m-cache-up-to-5-00-ghz/specifications.html)
- CPU: 8 Performance-cores, 4 Efficient-cores, 20 threads
  - Performance-core: 3.60 GHz base freq, turbo up to 4.90 GHz
  - Efficient-core: 2.70 GHz base freq, turbo up to 3.80 GHz

CPU: 

```
$ python3 benchmark.py --all
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
0.73       0.81       0.58       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
0.85       0.78       0.58       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
4.52       4.70       4.25       [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
6.67       7.25       4.25       [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
2.53       2.33       2.18       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
3.77       3.71       2.18       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
3.91       3.84       3.65       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
4.66       4.99       3.65       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
8.21       8.97       6.22       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
8.73       10.08      6.22       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
4.33       4.70       3.65       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
4.20       4.05       3.19       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
4.87       3.92       3.19       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
5.30       6.19       3.19       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
24.26      23.81      23.25      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
29.45      30.19      23.25      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
9.06       8.40       7.64       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
10.25      12.59      7.64       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
44.85      45.84      43.06      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
46.10      47.53      43.06      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
144.89     149.58     125.71     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
143.83     146.39     119.75     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
12.52      14.47      11.63      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
12.99      13.11      12.14      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
12.64      12.44      10.82      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
12.64      11.83      11.03      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
22.13      21.99      21.48      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
26.37      33.51      21.48      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
10.07      9.68       8.16       [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
1.19       1.30       1.07       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
80.97      80.06      73.20      [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
80.73      85.47      72.06      [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
17.97      16.18      12.43      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
19.54      20.66      12.43      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
17.73      24.25      9.65       [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
17.65      18.90      9.65       [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
16.97      15.14      9.65       [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
17.21      16.47      9.65       [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
17.68      14.54      9.65       [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
17.31      16.09      9.65       [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Rasberry Pi 4B

Specs: [details](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/)
- CPU: Broadcom BCM2711, Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5 GHz.

CPU:

```
$ python3 benchmark.py --all
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
5.96       5.93       5.90       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
6.09       6.11       5.90       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
73.30      73.22      72.32      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
88.20      89.95      72.32      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
32.33      32.20      31.99      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
39.82      40.78      31.99      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
108.37     108.31     106.93     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
75.91      78.95      49.78      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
76.29      77.10      75.21      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
77.33      77.73      75.21      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
66.22      66.09      65.90      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
59.91      60.72      54.63      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
62.83      54.85      54.63      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
62.47      62.13      54.63      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
625.82     667.05     425.55     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
508.92     667.04     373.14     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
147.19     146.62     146.31     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
143.70     155.87     139.90     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
214.87     214.19     213.21     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
212.90     212.93     209.55     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1690.06    2303.34    1480.63    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1489.54    1435.48    1308.12    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
90.49      89.23      86.83      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
356.63     357.29     354.42     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
217.52     229.39     101.61     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
198.63     198.25     196.68     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
417.23     434.54     388.38     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
381.72     394.15     308.62     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
194.47     195.18     191.67     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
5.90       5.90       5.81       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
2033.55    2454.13    1769.20    [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
1896.61    1977.38    1769.20    [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
237.73     237.57     236.82     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
265.16     270.22     236.82     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
239.69     298.68     198.88     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
234.90     249.29     198.88     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
227.47     200.42     198.88     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
226.39     213.26     198.88     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
226.10     227.18     198.88     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
220.63     217.04     193.47     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Jetson Nano B01

Specs: [details](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- CPU: Quad-core ARM A57 @ 1.43 GHz
- GPU: 128-core NVIDIA Maxwell

CPU:

```
$ python3 benchmark.py --all
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
5.64       5.55       5.50       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
5.91       6.00       5.50       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
61.32      61.38      61.08      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
76.85      78.69      61.08      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
27.39      27.54      27.26      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
34.69      35.62      27.26      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
50.39      50.31      50.22      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
48.97      49.42      47.46      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
68.07      67.81      67.72      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
73.97      74.83      67.72      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
63.85      63.63      63.51      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
55.14      55.93      47.84      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
60.80      48.09      47.84      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
60.99      61.22      47.84      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
352.73     352.51     351.53     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
374.22     376.71     351.53     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
134.60     135.00     133.68     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
137.10     137.32     133.68     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
215.10     215.30     214.30     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
216.18     216.19     214.30     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1207.83    1208.71    1203.64    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1236.98    1250.21    1203.64    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
123.30     125.37     116.69     [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
124.89     125.25     124.53     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
107.99     109.82     94.05      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
108.41     108.33     107.91     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
354.88     354.70     354.34     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
343.35     344.56     333.41     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
89.93      91.58      88.28      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
5.69       5.72       5.66       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
1070.55    1072.14    1055.67    [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
1071.56    1071.38    1055.67    [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
258.11     258.13     257.64     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
275.27     277.20     257.64     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
254.90     295.88     221.12     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
252.73     258.90     221.12     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
245.08     222.01     221.12     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
245.75     236.58     221.12     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
248.42     251.65     221.12     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
244.31     236.64     221.12     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

GPU (CUDA-FP32):
<!-- config wechat is excluded due to its api does not support setting backend and target -->
```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 1
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA
mean       median     min        input size   model
11.16      10.31      10.23      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
24.82      24.90      24.33      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
14.39      14.44      13.83      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
24.52      24.01      23.84      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
69.63      69.88      64.73      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
29.06      29.10      28.80      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
28.54      28.57      27.88      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
99.05      99.65      93.60      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
54.24      55.24      52.87      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
63.63      63.43      63.32      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
371.45     378.00     366.39     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
43.06      42.32      39.92      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
33.85      33.90      33.61      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
38.16      37.33      37.10      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
91.65      91.98      89.90      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
91.40      92.74      89.76      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
223.24     224.30     216.37     [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
223.03     222.28     216.37     [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
44.24      45.21      41.87      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
45.15      44.15      41.87      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
36.82      46.54      21.75      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

GPU (CUDA-FP16):
<!-- config wechat is excluded due to its api does not support setting backend and target -->
```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 2
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA_FP16
mean       median     min        input size   model
25.41      25.43      25.31      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
113.14     112.02     111.74     [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
89.04      88.90      88.59      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
96.62      96.39      96.26      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
69.78      70.65      66.74      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
118.47     118.45     118.10     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
125.69     126.63     118.10     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
64.08      62.97      62.33      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
366.46     366.88     363.46     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
163.06     163.34     161.77     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
301.10     311.52     297.74     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
43.36      40.65      39.85      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
149.37     149.95     148.01     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
153.89     153.96     153.43     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
44.29      44.03      43.62      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
91.28      92.89      89.79      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
254.78     256.13     245.60     [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
254.98     255.20     245.60     [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
33.07      32.88      32.00      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
33.88      33.64      32.00      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
29.32      33.70      20.69      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

### Khadas VIM3

Specs: [details](https://www.khadas.com/vim3)
- (SoC) CPU: Amlogic A311D, 2.2 GHz Quad core ARM Cortex-A73 and 1.8 GHz dual core Cortex-A53
- NPU: 5 TOPS Performance NPU INT8 inference up to 1536 MAC Supports all major deep learning frameworks including TensorFlow and Caffe 

CPU:

```
$ python3 benchmark.py --all
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
4.60       4.57       4.47       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
5.10       5.15       4.47       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
53.88      52.80      51.99      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
67.86      67.67      51.99      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
40.93      41.29      27.33      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
42.81      56.31      27.33      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
58.84      56.15      53.14      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
56.36      60.14      45.29      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
76.53      67.95      65.13      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
72.25      69.88      65.13      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
66.50      64.06      58.56      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
59.10      75.36      45.69      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
62.44      48.81      45.69      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
60.46      54.93      45.69      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
372.65     404.31     326.91     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
359.72     336.21     326.91     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
145.21     125.62     124.87     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
130.10     139.45     116.10     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
218.21     216.01     199.88     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
212.69     262.75     170.88     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1110.87    1112.27    1085.31    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1128.73    1157.12    1085.31    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
67.31      67.41      66.23      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
147.01     144.01     139.27     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
119.70     118.95     94.09      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
107.63     107.09     105.61     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
333.03     346.65     322.37     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
322.95     315.22     303.07     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
127.16     173.93     99.77      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
975.49     977.45     952.43     [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
970.16     970.83     928.66     [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
194.80     195.37     192.65     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
209.49     208.33     192.65     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
192.90     227.02     161.94     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
192.52     197.03     161.94     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
185.92     168.22     161.94     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
185.01     183.14     161.94     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
186.09     194.14     161.94     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
181.79     181.65     154.21     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

NPU (TIMVX):
<!-- config face_detection and licence_plate are excluded due to https://github.com/opencv/opencv_zoo/pull/190#discussion_r1257832066 -->
```
$ python3 benchmark.py --all --int8 --cfg_overwrite_backend_target 3 --cfg_exclude face_detection:license_plate
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_TIMVX
target=cv.dnn.DNN_TARGET_NPU
mean       median     min        input size   model
5.08       4.72       4.70       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
45.83      47.06      43.04      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
29.20      27.55      26.25      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
18.47      18.16      17.96      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
28.25      28.35      27.98      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
149.05     155.10     144.42     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
147.40     147.49     135.90     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
75.91      79.27      71.98      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
30.98      30.56      29.36      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
117.71     119.69     107.37     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
379.46     366.19     360.02     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
33.90      36.32      31.71      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
40.34      41.50      38.47      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
239.68     239.31     236.03     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
199.42     203.20     166.15     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
197.49     169.51     166.15     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Atlas 200 DK

Specs: [details_en](https://e.huawei.com/uk/products/cloud-computing-dc/atlas/atlas-200), [details_cn](https://www.hiascend.com/zh/hardware/developer-kit)
- (SoC) CPU: 8-core Coretext-A55 @ 1.6 GHz (max)
- NPU: Ascend 310, dual DaVinci AI cores, 22/16/8 TOPS INT8.

CPU:
<!-- config wechat is excluded due to it needs building with opencv_contrib -->
```
$ python3 benchmark.py --all --cfg_exclude wechat
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
7.82       7.82       7.77       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
8.57       8.77       7.77       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
92.21      92.11      91.87      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
122.07     126.02     91.87      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
42.93      43.26      42.75      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
55.91      57.40      42.75      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
67.85      67.91      67.47      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
70.06      70.21      67.47      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
102.49     102.65     102.10     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
114.02     116.16     102.10     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
92.66      92.49      92.36      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
79.39      80.75      68.47      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
89.66      68.66      68.47      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
90.59      92.13      68.47      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
499.55     500.15     498.36     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
571.85     580.88     498.36     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
201.99     201.55     200.62     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
216.72     217.34     200.62     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
313.66     313.85     312.13     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
322.98     323.45     312.13     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1875.33    1877.53    1871.26    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1989.04    2005.25    1871.26    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
143.62     143.19     137.16     [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
159.80     159.62     159.40     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
152.18     152.86     145.56     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
145.83     145.77     145.45     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
521.46     521.66     520.28     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
541.50     544.02     520.28     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
134.02     136.01     132.06     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
1441.73    1442.80    1440.26    [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
1436.45    1437.89    1430.58    [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
285.19     284.91     284.69     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
318.96     323.30     284.69     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
289.82     360.87     244.07     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
285.40     303.13     244.07     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
274.67     244.47     243.87     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
277.84     262.99     243.87     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
283.02     280.77     243.87     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
279.21     262.55     243.87     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

NPU (CANN):

<!-- vittrack is excluded due to HardSwish is not supported by CANN backend yet -->

```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat:crnn:vittrack --model_exclude pose_estimation_mediapipe_2023mar.onnx --cfg_overwrite_backend_target 4
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CANN
target=cv.dnn.DNN_TARGET_NPU
mean       median     min        input size   model
2.24       2.21       2.19       [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
2.66       2.66       2.64       [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
2.19       2.19       2.16       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
6.27       6.22       6.17       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
6.94       6.94       6.85       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
5.15       5.13       5.10       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
5.41       5.42       5.10       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
6.99       6.99       6.95       [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
7.63       7.64       7.43       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
20.62      22.09      19.16      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
28.59      28.60      27.91      [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
5.17       5.26       5.09       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
16.45      16.44      16.31      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
5.58       5.57       5.54       [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
17.15      17.18      16.83      [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
17.95      18.61      16.83      [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
```

### Toybrick RV1126

Specs: [details](https://t.rock-chips.com/en/portal.php?mod=view&aid=26)
- CPU: Quard core ARM Cortex-A7, up to 1.5GHz
- NPU (Not supported by OpenCV): 2.0TOPS, support 8bit / 16bit

CPU:
<!-- config wechat is excluded due to it needs building with opencv_contrib -->
```
$ python3 benchmark.py --all --cfg_exclude wechat
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
56.45      56.29      56.18      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
48.83      49.41      41.52      [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
1554.78    1545.63    1523.62    [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
1215.44    1251.08    921.26     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
612.58     613.61     587.83     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
502.02     513.29     399.51     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
525.72     532.34     502.00     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
415.87     442.23     318.14     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
1631.40    1635.83    1608.43    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
1115.29    1159.60    675.51     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
1546.54    1547.64    1516.69    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
1163.10    1227.05    816.99     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
980.56     852.38     689.31     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
837.72     778.61     507.03     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
11819.74   11778.79   11758.31   [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
7742.66    8151.17    4442.93    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
3266.08    3250.08    3216.03    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
2260.88    2368.00    1437.58    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
2335.65    2342.12    2304.69    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
1903.82    1962.71    1533.79    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
37604.10   37569.30   37502.48   [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
24229.20   25577.94   13483.54   [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
415.72     403.04     399.44     [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
1133.44    1131.54    1124.83    [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
883.96     919.07     655.33     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
1430.98    1424.55    1415.68    [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
11131.81   11141.37   11080.20   [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
7065.00    7461.37    3748.85    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
790.98     823.19     755.99     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
49331.32   49285.30   49210.67   [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
49327.34   49489.22   49210.67   [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
2183.70    2172.36    2156.29    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
2225.19    2222.58    2156.29    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
2214.03    2302.61    2156.29    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
2203.45    2231.47    2150.19    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
2201.14    2188.00    2150.19    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
2029.28    2178.36    1268.17    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
1923.12    2219.63    1268.17    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
1818.21    2196.98    1184.98    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Khadas Edge2 (with RK3588)

Board specs: [details](https://www.khadas.com/edge2)
SoC specs: [details](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html)
- CPU: 2.25GHz Quad Core ARM Cortex-A76 + 1.8GHz Quad Core Cortex-A55
- NPU (Not supported by OpenCV): Build-in 6 TOPS Performance NPU, triple core, support int4 / int8 / int16 / fp16 / bf16 / tf32

CPU:
<!-- config wechat is excluded due to it needs building with opencv_contrib -->
```
$ python3 benchmark.py --all --cfg_exclude wechat
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
2.29       2.30       2.25       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
2.62       2.64       2.25       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
28.19      28.12      28.01      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
36.68      37.80      28.01      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
12.56      12.55      12.50      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
17.28      17.83      12.50      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
22.74      22.87      22.43      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
24.56      24.61      22.43      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
29.91      30.23      28.16      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
35.54      35.46      28.16      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
27.28      27.20      27.20      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
22.91      23.33      19.28      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
27.36      19.46      19.28      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
28.28      29.17      19.28      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
150.06     150.89     147.05     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
180.91     184.75     147.05     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
54.14      52.95      49.31      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
60.01      61.20      49.31      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
117.60     128.98     83.33      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
117.28     150.31     83.33      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
553.58     558.76     535.47     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
594.18     592.64     535.47     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
49.47      49.21      48.84      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
56.35      55.73      55.25      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
57.07      57.19      55.25      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
47.94      48.41      47.05      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
146.02     145.89     139.08     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
157.60     158.88     139.08     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
41.26      42.74      40.08      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
384.47     401.25     360.71     [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
377.91     381.15     336.30     [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
68.70      68.63      68.54      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
78.17      80.48      68.54      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
71.42      91.44      61.14      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
70.07      76.28      61.14      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
67.69      61.72      61.14      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
68.29      65.04      61.14      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
69.58      68.63      61.14      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
68.99      65.02      61.14      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Horizon Sunrise X3 PI

Specs: [details_cn](https://developer.horizon.ai/sunrise)
- CPU: ARM Cortex-A53，4xCore, 1.2G
- BPU (aka NPU, not supported by OpenCV): (Bernoulli Arch) 2×Core，up to 1.0G, ~5Tops

CPU:

```
$ python3 benchmark.py --all
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
10.15      10.07      10.04      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
11.27      11.40      10.04      [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
116.44     116.29     116.15     [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
158.75     164.22     116.15     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
55.42      55.80      55.27      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
76.04      78.44      55.27      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
91.39      95.06      90.66      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
95.54      95.39      90.66      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
135.16     134.82     134.75     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
148.05     149.55     134.75     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
115.69     115.73     115.38     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
99.37      100.71     85.65      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
111.02     85.94      85.65      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
112.94     112.72     85.65      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
641.92     643.42     640.64     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
700.42     708.18     640.64     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
251.52     250.97     250.36     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
261.00     280.82     250.36     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
395.23     398.77     385.68     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
406.28     416.58     385.68     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
2608.90    2612.42    2597.93    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
2609.88    2609.39    2597.93    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
189.23     188.72     182.28     [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
228.95     228.74     228.35     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
227.97     228.61     226.76     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
192.29     192.26     191.74     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
660.62     662.28     659.49     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
646.25     647.89     631.03     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
182.57     185.52     179.71     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
9.93       9.97       9.82       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
1914.15    1913.70    1902.25    [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
1920.07    1929.80    1902.25    [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
439.96     441.91     436.49     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
465.56     466.86     436.49     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
431.93     495.94     373.61     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
432.47     435.40     373.61     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
418.75     375.76     373.61     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
421.81     410.25     373.61     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
429.30     437.71     373.61     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
422.15     406.50     373.61     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### MAIX-III AX-PI

Specs: [details_en](https://wiki.sipeed.com/hardware/en/maixIII/ax-pi/axpi.html#Hardware), [details_cn](https://wiki.sipeed.com/hardware/zh/maixIII/ax-pi/axpi.html#%E7%A1%AC%E4%BB%B6%E5%8F%82%E6%95%B0)
SoC specs: [details_cn](https://axera-tech.com/product/T7297367876123493768)
- CPU: Quad cores ARM Cortex-A7
- NPU (Not supported by OpenCV): 14.4Tops@int4，3.6Tops@int8

CPU:
<!-- config wechat is excluded due to it needs building with opencv_contrib -->
```
$ python3 benchmark.py --all --cfg_exclude wechat
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
83.67      83.60      83.50      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
76.45      77.17      70.53      [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
2102.93    2102.75    2102.23    [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
1846.25    1872.36    1639.46    [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
825.27     825.74     824.83     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
752.57     759.68     693.90     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
742.35     742.87     741.42     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
630.16     641.82     539.73     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
2190.53    2188.01    2187.75    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
1662.81    1712.08    1235.22    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
2099.43    2099.39    2098.89    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
1589.86    1641.45    1181.62    [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
1451.24    1182.16    1181.62    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
1277.21    1224.66    888.62     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
15832.31   15832.41   15830.59   [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
11649.30   12067.68   8300.79    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
4376.55    4398.44    4371.68    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
3376.78    3480.89    2574.72    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
3422.70    3414.45    3413.72    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
3002.36    3047.94    2655.38    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
50678.08   50651.82   50651.19   [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
36249.71   37771.22   24606.37   [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
707.79     706.32     699.40     [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
1502.15    1501.98    1500.99    [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
1300.15    1320.44    1137.60    [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
1993.05    1993.98    1991.86    [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
14925.56   14926.90   14912.28   [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
10507.96   10944.15   6974.74    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
1113.51    1124.83    1106.81    [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
66015.47   65997.60   65993.81   [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
66023.14   66034.99   65993.81   [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
3230.93    3228.61    3228.29    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
3312.02    3323.17    3228.29    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
3262.32    3413.03    3182.11    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
3250.66    3298.06    3182.11    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
3231.37    3185.37    3179.37    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
3064.17    3213.91    2345.80    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
2975.21    3227.38    2345.80    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
2862.33    3212.57    2205.48    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### StarFive VisionFive 2

Specs: [details_cn](https://doc.rvspace.org/VisionFive2/PB/VisionFive_2/specification_pb.html), [details_en](https://doc-en.rvspace.org/VisionFive2/Product_Brief/VisionFive_2/specification_pb.html)
- CPU: StarFive JH7110 with RISC-V quad-core CPU with 2 MB L2 cache and a monitor core, supporting RV64GC ISA, working up to 1.5 GHz
- GPU: IMG BXE-4-32 MC1 with work frequency up to 600 MHz

CPU:
<!-- config wechat is excluded due to it needs building with opencv_contrib -->
```
$ python3 benchmark.py --all --cfg_exclude wechat
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
41.10      41.09      41.04      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
35.87      36.37      31.62      [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
1050.45    1050.38    1050.01    [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
832.25     854.08     657.41     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
425.36     425.42     425.19     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
351.86     372.26     292.72     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
348.67     347.98     347.67     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
290.95     297.03     243.79     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
1135.09    1135.25    1134.72    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
788.33     822.69     509.67     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
1065.61    1065.99    1065.30    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
805.26     830.66     595.78     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
687.98     609.35     514.14     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
592.59     555.25     381.33     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
8091.50    8090.44    8088.72    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
5394.46    5666.14    3235.23    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
2270.14    2270.29    2267.51    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
1584.83    1656.13    1033.23    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
1732.53    1732.14    1731.47    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
1434.56    1463.32    1194.57    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
26172.62   26160.04   26151.67   [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
17004.06   17909.88   9659.54    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
304.58     309.56     280.05     [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
734.97     735.58     733.95     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
609.61     621.69     508.04     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
961.41     962.26     960.39     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
7594.21    7590.75    7589.16    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
4884.04    5154.38    2715.94    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
548.41     550.86     546.09     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
34074.19   34077.97   34058.43   [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
34073.67   34069.82   34054.29   [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
1397.09    1396.95    1396.74    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
1428.65    1432.59    1396.74    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
1429.56    1467.34    1396.74    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
1419.29    1450.55    1395.55    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
1421.72    1434.46    1395.55    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
1307.27    1415.63    807.66     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
1237.00    1395.68    807.66     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
1169.59    1415.29    774.09     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

<!--

### Khadas VIM4

CPU:

```
67.65      67.84      66.39      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
```

### NVIDIA Jetson Orin Nano

CPU:

```
59.30      58.45      57.90      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
```

CUDA:

```
13.69      13.69      13.04      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
```

CUDA-FP16:

```
16.29      15.77      15.77      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
```

### Atlas 200I DK

CPU:

```
88.24      87.00      84.23      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
```

-->
