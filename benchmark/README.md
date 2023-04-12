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

Benchmark is done with latest `opencv-python==4.7.0.72` and `opencv-contrib-python==4.7.0.72` on the following platforms. Some models are excluded because of support issues.

### Intel 12700K

Specs: [details](https://www.intel.com/content/www/us/en/products/sku/134594/intel-core-i712700k-processor-25m-cache-up-to-5-00-ghz/specifications.html)
- CPU: 8 Performance-cores, 4 Efficient-cores, 20 threads
  - Performance-core: 3.60 GHz base freq, turbo up to 4.90 GHz
  - Efficient-core: 2.70 GHz base freq, turbo up to 3.80 GHz

CPU: 

```
$ python benchmark.py --all --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
0.58       0.67       0.48       [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
0.82       0.81       0.48       [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
6.18       6.33       5.83       [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
7.42       7.42       5.83       [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
3.32       3.46       2.76       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
4.27       4.22       2.76       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
4.68       5.04       4.36       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
4.82       4.98       4.36       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
8.20       9.33       6.66       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
6.25       7.02       5.49       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
6.00       6.31       5.49       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
6.23       5.64       5.49       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
6.50       6.87       5.49       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
35.40      36.58      33.63      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
35.79      35.53      33.48      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
8.53       8.59       7.55       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
65.15      77.44      45.40      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
58.82      69.99      45.26      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
137.53     136.70     119.95     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
139.60     147.79     119.95     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
29.46      42.21      25.82      [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
6.14       6.02       5.91       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
8.51       9.89       5.91       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
30.87      30.69      29.85      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
30.77      30.02      27.97      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
1.35       1.37       1.30       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
75.82      75.37      69.18      [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
74.80      75.16      69.05      [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
21.37      24.50      16.04      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
23.08      25.14      16.04      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
20.43      31.14      11.74      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@145.253] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
20.71      17.95      11.74      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
19.48      25.14      11.74      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
19.38      18.85      11.74      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
19.52      25.97      11.74      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
18.55      15.29      10.35      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Rasberry Pi 4B

Specs: [details](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/)
- CPU: Broadcom BCM2711, Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5 GHz.

CPU:

```
$ python benchmark.py --all --model_exclude license_plate_detection_lpd_yunet_2023mar_int8.onnx:human_segmentation_pphumanseg_2023mar_int8.onnx
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
5.45       5.44       5.39       [160, 120]   YuNet with ['face_detection_yunet_2022mar.onnx']
6.12       6.15       5.39       [160, 120]   YuNet with ['face_detection_yunet_2022mar_int8.onnx']
78.04      77.96      77.62      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
91.44      93.03      77.62      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
32.21      31.86      31.85      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
38.22      39.27      31.85      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
43.85      43.76      43.51      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
46.66      47.00      43.51      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
73.29      73.70      72.86      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
74.51      87.71      73.83      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
67.29      68.22      61.55      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
68.53      61.77      61.55      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
68.31      72.16      61.55      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
547.70     547.68     494.91     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
527.14     567.06     465.02     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
192.61     194.08     156.62     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
248.03     229.41     209.65     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
246.41     247.64     207.91     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1932.97    1941.47    1859.96    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1866.98    1866.50    1746.67    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
762.56     738.04     654.25     [1280, 720]  DaSiamRPN with ['object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx', 'object_tracking_dasiamrpn_kernel_r1_2021nov.onnx', 'object_tracking_dasiamrpn_model_2021nov.onnx']
91.48      91.28      91.15      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
115.58     135.17     91.15      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
676.15     655.20     636.06     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
548.93     582.29     443.32     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
8.18       8.15       8.13       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
2025.09    2046.92    1971.57    [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
2041.85    2048.24    1971.57    [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
272.81     285.66     259.93     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
293.83     289.93     259.93     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
271.57     317.17     223.36     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
[ WARN:0@2039.612] global onnx_graph_simplifier.cpp:804 getMatFromTensor DNN: load FP16 model as FP32 model, and it takes twice the FP16 RAM requirement.
266.67     269.64     223.36     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
259.06     239.43     223.36     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
251.39     257.43     221.20     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
248.27     253.01     221.20     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
239.42     238.72     190.04     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
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
411.49     417.53     402.57     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
372.94     370.17     335.95     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
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
126.65     126.63     124.96     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
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
96.68      94.21      89.24      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
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
439.53     431.92     406.03     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
358.63     379.93     296.32     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
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
631.70     631.81     630.61     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
595.32     599.48     565.32     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
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
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat:dasiamrpn:crnn --cfg_overwrite_backend_target 4
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
5.58       5.57       5.54       [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
17.15      17.18      16.83      [640, 480]   DB with ['text_detection_DB_IC15_resnet18_2021sep.onnx']
17.95      18.61      16.83      [640, 480]   DB with ['text_detection_DB_TD500_resnet18_2021sep.onnx']
```
