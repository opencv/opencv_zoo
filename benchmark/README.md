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
python benchmark.py --all --cfg_exclude wechat:dasiamrpn

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
23.43      22.82      20.90      [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
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
564.90     580.35     527.49     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
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
456.79     456.90     445.83     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
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
77.50      77.73      76.16      [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
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
53.34      54.30      51.79      [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
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
382.57     464.42     354.66     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
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

```
$ python3 benchmark.py --all --int8 --cfg_overwrite_backend_target 3 --cfg_exclude face_detection:license_plate
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_TIMVX
target=cv.dnn.DNN_TARGET_NPU
mean       median     min        input size   model
45.83      47.06      43.04      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
29.20      27.55      26.25      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
18.47      18.16      17.96      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
28.25      28.35      27.98      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
149.05     155.10     144.42     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
147.40     147.49     135.90     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
75.91      79.27      71.98      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
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
637.54     640.61     626.98     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
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

```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat:dasiamrpn:crnn --model_exclude pose_estimation_mediapipe_2023mar.onnx --cfg_overwrite_backend_target 4
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

```
$ python3 benchmark.py --all --cfg_exclude wechat --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
68.89      68.59      68.23      [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
60.98      61.11      52.00      [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
1550.71    1578.99    1527.58    [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
1214.15    1261.66    920.50     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
604.36     611.24     578.99     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
496.42     537.75     397.23     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
460.56     470.15     440.77     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
387.63     379.96     318.71     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
1610.78    1599.92    1583.95    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
1546.16    1539.50    1513.14    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
1166.56    1211.97    827.10     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
983.80     868.18     689.32     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
840.38     801.83     504.54     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
11793.09   11817.73   11741.04   [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
7740.03    8134.99    4464.30    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
3222.92    3225.18    3170.71    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
2303.55    2307.46    2289.41    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
1888.15    1920.41    1528.78    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
38359.93   39021.21   37180.85   [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
24504.50   25439.34   13443.63   [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
14738.64   14764.84   14655.76   [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
872.09     877.72     838.99     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
764.48     775.55     653.25     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
1326.56    1327.10    1305.18    [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
11117.07   11109.12   11058.49   [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
7037.96    7424.89    3750.12    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
704.44     704.77     672.58     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
49065.03   49144.55   48943.50   [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
49052.24   48992.64   48927.44   [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
2200.08    2193.78    2175.77    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
2244.03    2240.25    2175.77    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
2230.12    2290.28    2175.77    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@1315.065] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
2220.33    2281.75    2171.61    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
2216.44    2212.48    2171.61    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
2041.65    2209.50    1268.91    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
1933.06    2210.81    1268.91    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
1826.34    2234.66    1184.53    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Khadas Edge2 (with RK3588)

Board specs: [details](https://www.khadas.com/edge2)
SoC specs: [details](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html)
- CPU: 2.25GHz Quad Core ARM Cortex-A76 + 1.8GHz Quad Core Cortex-A55
- NPU (Not supported by OpenCV): Build-in 6 TOPS Performance NPU, triple core, support int4 / int8 / int16 / fp16 / bf16 / tf32

CPU:

```
$ python3 benchmark.py --all --cfg_exclude wechat --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
2.47       2.55       2.44       [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
2.81       2.84       2.44       [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
33.79      33.83      33.24      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
39.96      40.77      33.24      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
15.99      16.12      15.92      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
19.09      19.48      15.92      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
20.27      20.45      20.11      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
23.14      23.62      20.11      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
34.58      34.53      33.55      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
32.78      32.94      31.99      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
28.38      28.80      24.59      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
31.49      24.66      24.59      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
31.45      32.34      24.59      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
178.87     178.49     173.57     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
197.19     200.06     173.57     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
57.57      65.48      51.34      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
118.38     132.59     88.34      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
120.74     110.82     88.34      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
577.93     577.17     553.81     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
607.96     604.88     553.81     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
152.78     155.89     121.26     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
38.03      38.26      37.51      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
47.12      48.12      37.51      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
46.07      46.77      45.10      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
195.67     198.02     182.97     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
181.91     182.28     169.98     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
35.47      37.63      33.55      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
394.77     407.60     371.95     [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
392.52     404.80     367.96     [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
77.32      77.72      75.27      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
82.93      82.93      75.27      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
77.51      93.01      67.44      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@598.857] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
77.02      84.11      67.44      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
75.11      69.82      63.98      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
74.55      73.36      63.98      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
75.06      77.44      63.98      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
73.91      74.25      63.98      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
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
809.55     814.66     794.67     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
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

```
$ python3 benchmark.py --all --cfg_exclude wechat --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
98.16      98.99      97.73      [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
93.21      93.81      89.15      [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
2093.12    2093.02    2092.54    [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
1845.87    1871.17    1646.65    [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
811.32     811.47     810.80     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
743.24     750.04     688.44     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
636.22     635.89     635.43     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
588.83     594.01     550.49     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
2157.86    2157.82    2156.99    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
2091.13    2091.61    2090.72    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
1583.25    1634.14    1176.19    [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
1450.55    1177.07    1176.19    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
1272.81    1226.00    873.94     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
15753.56   15751.29   15748.97   [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
11610.11   12023.99   8290.04    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
4300.13    4301.43    4298.29    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
3360.20    3357.84    3356.70    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
2961.58    3005.40    2641.27    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
49994.75   49968.90   49958.48   [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
35966.66   37391.40   24670.30   [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
19800.14   19816.02   19754.69   [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
1191.81    1192.42    1191.40    [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
1162.64    1165.77    1138.35    [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
1835.97    1836.24    1835.34    [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
14886.02   14884.48   14881.73   [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
10491.63   10930.80   6975.34    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
987.30     992.59     982.71     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
65681.91   65674.89   65612.09   [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
65630.56   65652.90   65531.21   [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
3248.11    3242.59    3241.18    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
3330.69    3350.38    3241.18    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
3277.07    3427.65    3195.84    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@17240.397] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
3263.48    3319.83    3195.84    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
3258.78    3196.90    3195.84    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
3090.12    3224.64    2353.81    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
3001.31    3237.93    2353.81    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
2887.05    3224.12    2206.89    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### StarFive VisionFive 2

Specs: [details_cn](https://doc.rvspace.org/VisionFive2/PB/VisionFive_2/specification_pb.html), [details_en](https://doc-en.rvspace.org/VisionFive2/Product_Brief/VisionFive_2/specification_pb.html)
- CPU: StarFive JH7110 with RISC-V quad-core CPU with 2 MB L2 cache and a monitor core, supporting RV64GC ISA, working up to 1.5 GHz
- GPU: IMG BXE-4-32 MC1 with work frequency up to 600 MHz

CPU:

```
$ python3 benchmark.py --all --cfg_exclude wechat:dasiam --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
50.28      50.42      50.08      [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
44.45      44.84      39.29      [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
1059.87    1059.79    1058.95    [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
838.07     859.42     658.86     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
424.55     424.74     424.06     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
350.30     357.95     290.66     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
314.50     313.75     313.67     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
275.80     280.48     243.97     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
1131.91    1132.16    1131.08    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
1072.77    1073.31    1072.07    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
811.64     837.32     602.08     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
692.68     602.74     516.39     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
596.12     559.52     382.75     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
8131.86    8132.90    8128.55    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
5412.98    5684.12    3236.35    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
2265.62    2264.83    2263.38    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
1727.39    1727.31    1726.31    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
1429.48    1458.69    1189.19    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
26156.87   26169.88   26134.95   [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
17151.71   17933.90   9675.03    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
316.26     315.72     315.55     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
276.38     280.84     243.11     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
586.18     586.28     585.62     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
542.79     546.26     506.12     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
910.67     910.62     909.72     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
7628.31    7624.65    7623.26    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
4899.76    5171.88    2714.07    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
486.59     490.33     484.31     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
34888.37   34834.51   34103.30   [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
35123.00   35996.09   34103.30   [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
1425.08    1543.33    1413.01    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
1455.55    1580.51    1413.01    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
1457.01    1484.13    1413.01    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
1281.84    1468.77    810.51     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
1191.52    1517.48    810.51     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
1111.95    1131.27    775.96     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```
