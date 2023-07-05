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
$ python3 benchmark.py --all --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
5.37       5.44       5.27       [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
6.11       7.99       5.27       [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
65.14      65.13      64.93      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
79.33      88.12      64.93      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
28.19      28.17      28.05      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
34.85      35.66      28.05      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
41.02      42.37      40.80      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
44.20      44.39      40.80      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
65.91      65.93      65.68      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
68.94      68.95      68.77      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
62.12      62.24      55.29      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
66.04      55.58      55.29      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
65.31      64.86      55.29      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
376.88     368.22     367.11     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
390.32     385.28     367.11     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
133.15     130.57     129.38     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
215.57     225.11     212.66     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
217.37     214.85     212.66     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1228.13    1233.90    1219.11    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1257.34    1256.26    1219.11    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
466.19     457.89     442.88     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
69.60      69.69      69.13      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
81.65      82.20      69.13      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
98.38      98.20      97.69      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
411.49     417.53     402.57     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
372.94     370.17     335.95     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
74.36      75.15      72.22      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
5.62       5.64       5.55       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
1089.89    1091.85    1071.95    [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
1089.94    1095.07    1071.95    [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
274.45     286.03     270.52     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
290.82     288.87     270.52     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
269.52     311.59     228.47     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@1497.159] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
269.66     267.98     228.47     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
261.39     231.92     228.47     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
259.68     249.43     228.47     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
260.89     283.44     228.47     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
255.61     249.41     222.38     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

GPU (CUDA-FP32):
```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 1
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA
mean       median     min        input size   model
11.22      11.49      9.59       [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
24.60      25.91      24.16      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
20.64      24.00      18.88      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
41.15      41.18      40.95      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
90.86      90.79      84.96      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
69.24      69.11      68.87      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
62.12      62.30      55.28      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
148.58     153.17     144.61     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
53.50      54.29      51.48      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
214.99     218.04     212.94     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
1238.91    1244.87    1227.30    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
76.54      76.09      74.51      [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
67.34      67.83      62.38      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
56.69      55.54      48.96      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
126.65     126.63     124.96     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
73.84      75.25      72.19      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
303.12     302.80     299.30     [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
302.58     299.78     297.83     [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
58.05      62.90      52.47      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
59.39      56.82      52.47      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
45.60      62.40      21.73      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

GPU (CUDA-FP16):

```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 2
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA_FP16
mean       median     min        input size   model
26.17      26.40      25.87      [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
116.07     115.93     112.39     [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
119.85     121.62     114.63     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
40.94      40.92      40.70      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
99.88      100.49     93.24      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
69.00      68.81      68.60      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
61.93      62.18      55.17      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
141.11     145.82     136.02     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
364.70     363.48     360.28     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
215.23     213.49     213.06     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
1223.32    1248.88    1213.25    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
52.91      52.96      50.17      [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
212.86     213.21     210.03     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
221.12     255.53     217.16     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
96.68      94.21      89.24      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
73.68      77.30      69.17      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
343.38     344.17     337.62     [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
344.29     345.07     337.62     [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
48.91      50.31      45.41      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
50.20      49.66      45.41      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
39.56      52.56      20.76      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

### Khadas VIM3

Specs: [details](https://www.khadas.com/vim3)
- (SoC) CPU: Amlogic A311D, 2.2 GHz Quad core ARM Cortex-A73 and 1.8 GHz dual core Cortex-A53
- NPU: 5 TOPS Performance NPU INT8 inference up to 1536 MAC Supports all major deep learning frameworks including TensorFlow and Caffe 

CPU:

```
$ python3 benchmark.py --all --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
4.93       4.91       4.83       [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
5.30       5.31       4.83       [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
60.02      61.00      57.85      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
70.27      74.77      57.85      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
29.36      28.28      27.97      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
34.66      34.12      27.97      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
38.60      37.72      36.79      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
41.57      41.91      36.79      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
70.82      72.70      67.14      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
64.73      64.22      62.19      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
58.18      59.29      49.97      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
59.15      52.27      49.97      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
57.38      55.13      49.97      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
385.29     361.27     348.96     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
352.90     395.79     328.06     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
122.17     123.58     119.43     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
208.25     217.96     195.76     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
203.04     213.99     161.37     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1189.83    1150.85    1138.93    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1137.18    1142.89    1080.23    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
428.66     524.98     391.33     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
66.91      67.09      64.90      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
79.42      81.44      64.90      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
84.42      85.99      83.30      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
439.53     431.92     406.03     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
358.63     379.93     296.32     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
68.51      66.87      66.53      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
5.29       5.30       5.21       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
973.75     968.68     954.58     [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
961.44     959.29     935.29     [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
202.74     202.73     200.75     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
217.07     217.26     200.75     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
199.81     231.31     169.27     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@1277.652] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
199.73     203.96     169.27     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
192.97     175.68     169.27     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
189.65     189.43     169.27     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
188.98     202.49     169.27     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
183.49     188.71     149.81     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

NPU (TIMVX):

```
$ python3 benchmark.py --all --int8 --cfg_overwrite_backend_target 3 --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_TIMVX
target=cv.dnn.DNN_TARGET_NPU
mean       median     min        input size   model
5.67       5.74       5.59       [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
76.97      77.86      75.59      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
40.38      39.41      38.12      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
44.36      45.77      42.06      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
60.75      62.46      56.34      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
57.40      58.10      52.11      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
340.20     347.74     330.70     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
200.50     224.02     160.81     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1103.24    1091.76    1059.77    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
95.92      102.80     92.77      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
307.90     310.52     302.46     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
178.71     178.87     177.84     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
183.51     183.72     177.84     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
172.06     189.19     149.19     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Atlas 200 DK

Specs: [details_en](https://e.huawei.com/uk/products/cloud-computing-dc/atlas/atlas-200), [details_cn](https://www.hiascend.com/zh/hardware/developer-kit)
- (SoC) CPU: 8-core Coretext-A55 @ 1.6 GHz (max)
- NPU: Ascend 310, dual DaVinci AI cores, 22/16/8 TOPS INT8.

CPU:

```
$ python3 benchmark.py --all --cfg_exclude wechat --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
8.02       8.07       7.93       [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
9.44       9.34       7.93       [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
104.51     112.90     102.07     [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
131.49     147.17     102.07     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
47.71      57.86      46.48      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
59.26      59.07      46.48      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
57.95      58.02      57.30      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
65.52      70.76      57.30      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
107.98     127.65     106.59     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
103.96     124.91     102.87     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
90.46      90.53      76.14      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
98.40      76.49      76.14      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
98.06      95.36      76.14      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
564.69     556.79     537.84     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
621.54     661.56     537.84     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
226.08     216.89     216.07     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
343.08     346.39     315.99     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
351.64     346.41     315.99     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1995.97    1996.82    1967.76    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
2060.87    2055.60    1967.76    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
701.08     708.52     685.49     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
105.23     105.14     105.00     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
123.41     125.65     105.00     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
134.10     134.43     133.62     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
631.70     631.81     630.61     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
595.32     599.48     565.32     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
108.55     117.88     106.66     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
1452.55    1453.75    1450.98    [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
1433.26    1432.08    1409.78    [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
299.36     299.92     298.75     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
329.84     333.32     298.75     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
303.65     367.68     262.48     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@760.743] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
299.60     315.91     262.48     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
290.29     263.05     262.48     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
290.41     279.30     262.48     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
294.61     295.36     262.48     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
289.53     279.60     262.48     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

NPU:

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
$ python3 benchmark.py --all --cfg_exclude wechat --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
11.04      11.01      10.98      [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
12.59      12.75      10.98      [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
140.83     140.85     140.52     [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
171.71     175.65     140.52     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
64.96      64.94      64.77      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
80.20      81.82      64.77      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
80.67      80.72      80.45      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
89.25      90.39      80.45      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
144.23     144.34     143.84     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
140.60     140.62     140.33     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
122.53     124.23     107.71     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
128.22     107.87     107.71     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
125.77     123.77     107.71     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
759.81     760.01     759.11     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
764.17     764.43     759.11     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
283.75     284.17     282.15     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
408.16     408.31     402.71     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
408.82     407.99     402.71     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
2749.22    2756.23    2737.96    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
2671.54    2692.18    2601.24    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
929.63     936.01     914.86     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
142.23     142.03     141.78     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
179.74     184.79     141.78     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
191.41     191.48     191.00     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
898.23     897.52     896.58     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
749.83     765.90     630.39     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
158.50     160.55     155.64     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
1908.87    1905.00    1903.13    [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
1922.34    1920.65    1896.97    [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
470.78     469.17     467.92     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
495.94     497.12     467.92     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
464.58     528.72     408.69     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@2820.735] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
465.04     467.01     408.69     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
452.90     409.34     408.69     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
450.23     438.57     408.69     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
453.52     468.72     408.69     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
443.38     447.29     381.90     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
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
