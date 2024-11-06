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

# All configs but only fp32 models (--fp32, --fp16, --int8 --int8bq are available for now)
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

Benchmark is done with latest opencv-python & opencv-contrib-python (current 4.10.0) on the following platforms. Some models are excluded because of support issues.

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
0.69       0.70       0.68       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
0.79       0.80       0.68       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
5.09       5.13       4.96       [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
6.50       6.79       4.96       [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
1.79       1.76       1.75       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
2.92       3.11       1.75       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
2.40       2.43       2.37       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
3.11       3.15       2.37       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
5.59       5.56       5.28       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
6.07       6.22       5.28       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
3.13       3.14       3.05       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
3.04       3.02       2.92       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
3.46       3.03       2.92       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
3.84       3.77       2.92       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
19.47      19.47      19.08      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
21.52      21.86      19.08      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
5.68       5.66       5.51       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
7.41       7.36       5.51       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
41.02      40.99      40.86      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
42.23      42.30      40.86      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
78.77      79.76      77.16      [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
75.69      75.58      72.57      [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
4.01       3.84       3.79       [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
5.35       5.41       5.22       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
6.73       6.85       5.22       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
7.65       7.65       7.55       [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
15.56      15.57      15.10      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
16.67      16.57      15.10      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
6.33       6.63       6.14       [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
1.19       1.30       1.07       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
18.76      19.59      18.48      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
18.59      19.33      18.12      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
22.05      18.60      18.12      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
24.47      25.06      18.12      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
10.61      10.66      10.50      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
11.03      11.23      10.50      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
9.85       11.62      7.74       [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
10.02      9.71       7.74       [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
9.53       7.83       7.74       [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
9.68       9.21       7.74       [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
9.85       10.63      7.74       [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
9.63       9.28       7.74       [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Raspberry Pi 4B

Specs: [details](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/)
- CPU: Broadcom BCM2711, Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5 GHz.

CPU:

```
$ python3 benchmark.py --all
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
6.23       6.27       6.18       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
6.68       6.73       6.18       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
68.82      69.06      68.45      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
87.42      89.84      68.45      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
27.81      27.77      27.67      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
35.71      36.67      27.67      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
42.58      42.41      42.25      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
46.49      46.95      42.25      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
71.35      71.62      70.78      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
73.81      74.23      70.78      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
64.20      64.30      63.98      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
57.91      58.41      52.53      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
61.35      52.83      52.53      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
61.49      61.28      52.53      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
420.93     420.73     419.04     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
410.96     395.74     364.68     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
153.87     152.71     140.85     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
157.86     145.90     140.85     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
214.59     211.95     210.98     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
215.09     238.39     208.18     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1614.13    1639.80    1476.58    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1597.92    1599.12    1476.58    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
48.55      46.87      41.75      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
97.05      95.40      80.93      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
112.39     116.22     80.93      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
105.60     113.27     88.55      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
478.89     498.05     444.14     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
442.56     477.87     369.59     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
116.15     120.13     106.81     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
5.90       5.90       5.81       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
325.02     325.88     303.55     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
323.54     332.45     303.55     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
372.32     328.56     303.55     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
407.90     411.97     303.55     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
235.70     236.07     234.87     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
240.95     241.14     234.87     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
226.09     247.02     200.44     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
229.25     224.63     200.44     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
224.10     201.29     200.44     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
223.58     219.82     200.44     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
225.60     243.89     200.44     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
220.97     223.16     193.91     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
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
5.62       5.54       5.52       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
6.14       6.24       5.52       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
64.80      64.95      64.60      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
78.31      79.85      64.60      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
26.54      26.61      26.37      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
33.96      34.85      26.37      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
38.45      41.45      38.20      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
42.62      43.20      38.20      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
64.95      64.85      64.73      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
72.39      73.16      64.73      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
65.72      65.98      65.59      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
56.66      57.56      49.10      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
62.09      49.27      49.10      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
62.17      62.02      49.10      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
346.78     348.06     345.53     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
371.11     373.54     345.53     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
134.36     134.33     133.45     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
140.62     140.94     133.45     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
215.67     216.76     214.69     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
216.58     216.78     214.69     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1209.12    1213.05    1201.68    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1240.02    1249.95    1201.68    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
48.39      47.38      45.00      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
75.30      75.25      74.96      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
83.83      84.99      74.96      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
87.65      87.59      87.37      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
356.78     357.77     355.69     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
346.84     351.10     335.96     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
75.20      79.36      73.71      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
5.56       5.56       5.48       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
209.80     210.04     208.84     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
209.60     212.74     208.49     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
254.56     211.17     208.49     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
286.57     296.56     208.49     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
252.60     252.48     252.21     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
259.28     261.38     252.21     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
245.18     266.94     220.49     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
247.72     244.25     220.49     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
241.63     221.43     219.06     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
243.46     238.98     219.06     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
246.87     256.05     219.06     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
243.37     238.90     219.06     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

GPU (CUDA-FP32):
<!-- config wechat is excluded due to its api does not support setting backend and target -->
```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 1
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA
mean       median     min        input size   model
10.99      10.71      9.64       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
25.25      25.81      24.54      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
13.97      14.01      13.72      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
24.47      24.36      23.69      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
67.25      67.99      64.90      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
28.96      28.92      28.85      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
28.61      28.45      27.92      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
98.80      100.11     94.57      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
54.88      56.51      52.78      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
63.86      63.59      63.35      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
371.32     374.79     367.78     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
47.26      45.56      44.69      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
37.61      37.61      33.64      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
37.39      37.71      37.03      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
90.84      91.34      85.77      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
76.44      78.00      74.90      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
112.68     112.21     110.42     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
112.48     111.86     110.04     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
43.99      43.33      41.68      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
44.97      44.42      41.68      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
36.77      46.38      21.77      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

GPU (CUDA-FP16):
<!-- config wechat is excluded due to its api does not support setting backend and target -->
```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 2
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA_FP16
mean       median     min        input size   model
25.05      25.05      24.95      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
117.82     126.96     113.17     [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
88.54      88.33      88.04      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
97.43      97.38      96.98      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
69.40      68.28      66.36      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
120.92     131.57     119.37     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
128.43     128.08     119.37     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
64.90      63.88      62.81      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
370.21     371.97     366.38     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
164.28     164.75     162.94     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
299.22     300.54     295.64     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
49.61      47.58      47.14      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
149.50     151.12     147.24     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
156.59     154.01     153.92     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
43.66      43.64      43.31      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
75.87      77.33      74.38      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
428.97     428.99     426.11     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
428.66     427.46     425.66     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
32.41      31.90      31.68      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
33.42      35.75      31.68      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
29.34      36.44      21.27      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

### Khadas VIM3

Specs: [details](https://www.khadas.com/vim3)
- (SoC) CPU: Amlogic A311D, 2.2 GHz Quad core ARM Cortex-A73 and 1.8 GHz dual core Cortex-A53
- NPU: 5 TOPS Performance NPU INT8 inference up to 1536 MAC Supports all major deep learning frameworks including TensorFlow and Caffe 

CPU:
<!-- config wechat is excluded due to it needs building with opencv_contrib -->
```
$ python3 benchmark.py --all --cfg_exclude wechat
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
4.62       4.62       4.53       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
5.24       5.29       4.53       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
55.04      54.55      53.54      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
67.34      67.96      53.54      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
29.50      45.62      26.14      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
35.59      36.22      26.14      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
35.80      35.08      34.76      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
40.32      45.32      34.76      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
71.92      66.92      62.98      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
70.68      72.31      62.98      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
59.27      53.91      52.09      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
52.17      67.58      41.23      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
55.44      47.28      41.23      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
55.83      56.80      41.23      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
335.75     329.39     325.42     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
340.42     335.78     325.42     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
128.58     127.15     124.03     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
125.85     126.47     110.14     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
179.93     170.66     166.76     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
178.61     213.72     164.61     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1108.12    1100.93    1072.45    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1100.58    1121.31    982.74     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
32.20      32.84      30.99      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
78.26      78.96      75.60      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
87.18      88.22      75.60      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
83.22      84.20      80.07      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
327.07     339.80     321.98     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
316.56     302.60     269.10     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
75.38      73.67      70.15      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
211.02     213.14     199.28     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
210.19     217.15     199.28     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
242.34     225.59     199.28     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
265.33     271.87     199.28     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
194.77     195.13     192.69     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
197.16     200.94     192.69     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
185.45     199.47     161.37     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
187.64     180.57     161.37     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
182.53     166.96     161.37     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
182.90     178.97     161.37     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
184.26     194.43     161.37     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
180.65     180.59     155.36     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

NPU (TIMVX):

```
$ python3 benchmark.py --all --int8 --cfg_overwrite_backend_target 3
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_TIMVX
target=cv.dnn.DNN_TARGET_NPU
mean       median     min        input size   model
5.24       7.45       4.77       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
45.96      46.10      43.21      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
30.25      30.30      28.68      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
19.75      20.18      18.19      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
28.75      28.85      28.47      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
148.80     148.85     143.45     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
143.17     141.11     136.58     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
73.19      78.57      62.89      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
32.11      30.50      29.97      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
116.32     120.72     99.40      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
408.18     418.89     374.12     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
37.34      38.57      32.03      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
41.82      39.84      37.63      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
160.70     160.90     153.15     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
160.47     160.48     151.88     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
239.38     237.47     231.95     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
197.61     201.16     162.69     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
196.69     164.78     162.69     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
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
360.26     360.82     359.13     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
361.22     361.51     359.13     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
427.85     362.87     359.13     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
475.44     490.06     359.13     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
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
56.78      56.74      56.46      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
51.16      51.41      45.18      [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
1737.74    1733.23    1723.65    [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
1298.48    1336.02    920.44     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
609.51     611.79     584.89     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
500.21     517.38     399.97     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
465.12     471.89     445.36     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
389.95     385.01     318.29     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
10.16.66.1781623.94    1607.90    1595.09    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
1109.61    1186.03    671.15     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
1567.09    1578.61    1542.75    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
1188.83    1219.46    850.92     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
996.30     884.80     689.11     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
849.51     805.93     507.78     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
11855.64   11836.80   11750.10   [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
7752.60    8149.00    4429.83    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
3260.22    3251.14    3204.85    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
2287.10    2400.53    1482.04    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
2335.89    2335.93    2313.63    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
1899.16    1945.72    1529.46    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
37600.81   37558.85   37414.98   [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
24185.35   25519.27   13395.47   [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
411.41     448.29     397.86     [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
905.77     890.22     866.06     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
780.94     817.69     653.26     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
1315.48    1321.44    1299.68    [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
11143.23   11155.05   11105.11   [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
7056.60    7457.76    3753.42    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
736.02     732.90     701.14     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
4267.03    4288.42    4229.69    [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
4265.58    4276.54    4222.22    [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
3678.65    4265.95    2636.57    [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
3383.73    3490.66    2636.57    [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
2180.44    2197.45    2152.67    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
2217.08    2241.77    2152.67    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
2217.15    2251.65    2152.67    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
2206.73    2219.60    2152.63    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
2208.84    2219.14    2152.63    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
2035.98    2185.05    1268.94    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
1927.93    2178.84    1268.94    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
1822.23    2213.30    1183.93    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
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
2.30       2.29       2.26       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
2.70       2.73       2.26       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
28.94      29.00      28.60      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
37.46      38.85      28.60      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
12.44      12.40      12.36      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
17.14      17.64      12.36      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
20.22      20.36      20.08      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
23.11      23.50      20.08      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
29.63      29.78      28.61      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
35.57      35.61      28.61      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
27.45      27.46      27.25      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
22.95      23.37      19.13      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
27.50      19.40      19.13      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
28.46      29.33      19.13      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
151.10     151.79     146.96     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
181.69     184.19     146.96     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
53.83      52.64      50.24      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
60.95      60.06      50.24      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
98.03      104.53     83.47      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
106.91     110.68     83.47      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
554.30     550.32     538.99     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
591.95     599.62     538.99     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
14.02      13.89      13.56      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
45.03      44.65      43.28      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
50.87      52.24      43.28      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
42.90      42.68      42.40      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
148.01     146.42     139.56     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
159.16     155.98     139.56     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
37.06      37.43      36.39      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
103.42     104.24     101.26     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
103.41     104.41     100.08     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
126.21     103.90     100.08     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
142.53     147.66     100.08     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
69.49      69.52      69.17      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
70.63      70.69      69.17      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
67.15      72.03      61.13      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
67.74      66.72      61.13      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
66.26      61.46      61.13      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
67.36      65.65      61.13      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
68.52      69.93      61.13      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
68.36      65.65      61.13      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
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
10.56      10.69      10.46      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
12.45      12.60      10.46      [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
124.80     127.36     124.45     [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
168.67     174.03     124.45     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
55.12      55.38      54.91      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
76.31      79.00      54.91      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
77.44      77.53      77.07      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
89.22      90.40      77.07      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
132.95     133.21     132.35     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
147.40     149.99     132.35     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
119.71     120.69     119.32     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
102.57     104.40     88.49      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
114.56     88.81      88.49      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
117.12     116.07     88.49      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
653.39     653.85     651.99     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
706.43     712.61     651.99     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
252.05     252.16     250.98     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
273.03     274.27     250.98     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
399.35     405.40     390.82     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
413.37     410.75     390.82     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
2516.91    2516.82    2506.54    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
2544.65    2551.55    2506.54    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
84.15      85.18      77.31      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
168.54     169.05     168.15     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
196.46     199.81     168.15     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
172.55     172.83     171.85     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
678.74     678.04     677.44     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
653.71     655.74     631.68     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
162.87     165.82     160.04     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
9.93       9.97       9.82       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
475.98     475.34     472.72     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
475.90     477.57     472.44     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
585.72     475.98     472.44     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
663.34     687.10     472.44     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
446.82     445.92     444.32     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
453.60     456.07     444.32     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
427.47     463.88     381.10     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
432.15     421.18     381.10     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
420.61     386.28     380.35     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
425.24     426.69     380.35     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
431.14     447.85     380.35     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
424.77     417.01     380.35     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
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
83.95      83.76      83.62      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
79.35      79.92      75.47      [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
2326.96    2326.49    2326.08    [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
1950.83    1988.86    1648.47    [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
823.42     823.35     822.50     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
750.31     757.91     691.41     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
664.73     664.61     663.84     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
596.29     603.96     540.72     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
2175.34    2173.62    2172.91    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
1655.11    1705.43    1236.22    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
2123.08    2122.92    2122.18    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
1619.08    1672.32    1215.05    [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
1470.74    1216.86    1215.05    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
1287.09    1242.01    873.92     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
15841.89   15841.20   15828.32   [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
11652.03   12079.50   8299.15    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
4371.75    4396.81    4370.29    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
3428.89    3521.87    2670.46    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
3421.19    3412.22    3411.20    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
2990.22    3034.11    2645.09    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
50633.38   50617.44   50614.78   [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
36260.23   37731.28   24683.40   [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
548.36     551.97     537.90     [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
1285.54    1285.40    1284.43    [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
1204.04    1211.89    1137.65    [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
1849.87    1848.78    1847.80    [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
14895.99   14894.27   14884.17   [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
10496.44   10931.97   6976.60    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
1045.98    1052.05    1040.56    [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
5899.23    5900.08    5896.73    [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
5889.39    5890.58    5878.81    [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
5436.61    5884.03    4665.77    [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
5185.53    5273.76    4539.47    [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
3230.95    3226.14    3225.53    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
3281.31    3295.46    3225.53    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
3247.56    3337.52    3196.25    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
3243.20    3276.35    3196.25    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
3230.49    3196.80    3195.02    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
3065.33    3217.99    2348.42    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
2976.24    3244.75    2348.42    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
2864.72    3219.46    2208.44    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
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
41.13      41.07      41.06      [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
37.43      37.83      34.35      [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
1169.96    1169.72    1168.74    [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
887.13     987.00     659.71     [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
423.91     423.98     423.62     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
350.89     358.26     292.27     [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
319.69     319.26     318.76     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
278.74     282.75     245.22     [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
1127.61    1127.36    1127.17    [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
785.44     819.07     510.77     [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
1079.69    1079.66    1079.31    [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
820.15     845.54     611.26     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
698.13     612.64     516.41     [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
600.12     564.13     382.59     [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
8116.21    8127.96    8113.70    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
5408.02    5677.71    3240.16    [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
2267.96    2268.26    2266.59    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
1605.80    1671.91    1073.50    [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
1731.61    1733.17    1730.54    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
1435.43    1477.52    1196.01    [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
26185.41   26190.85   26168.68   [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
17019.14   17923.20   9673.68    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
288.95     290.28     260.40     [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
628.64     628.47     628.27     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
562.90     569.91     509.93     [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
910.38     910.94     909.64     [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
7613.64    7626.26    7606.07    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
4895.28    5166.85    2716.65    [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
524.52     526.33     522.71     [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
2988.22    2996.51    2980.17    [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
2981.84    2979.74    2975.80    [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
2610.78    2979.14    1979.37    [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
2425.29    2478.92    1979.37    [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
1404.01    1415.46    1401.36    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
1425.42    1426.51    1401.36    [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
1432.21    1450.47    1401.36    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
1425.24    1448.27    1401.36    [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
1428.84    1446.76    1401.36    [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
1313.68    1427.46    808.70     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
1242.07    1408.93    808.70     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
1174.32    1426.07    774.78     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Khadas VIM4

Board specs: https://www.khadas.com/vim4, https://dl.khadas.com/products/vim4/specs/vim4-specs.pdf

SoC specs:
- CPU: Amlogic A311D2, 2.2GHz Quad core ARM Cortex-A73 and 2.0GHz Quad core Cortex-A53 CPU, with 32-bit STM32G031K6 microprocessor.
- GPU: Mali G52MP8(8EE) 800Mhz GPU.
- NPU: 3.2 TOPS Build-in NPU (Not supported by dnn yet)

CPU:
<!-- config wechat is excluded due to it needs building with opencv_contrib -->
```
$ python3 benchmark.py --all --cfg_exclude wechat
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
4.27       4.33       4.17       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
4.58       4.58       4.17       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
39.94      39.98      39.42      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
49.33      50.59      39.42      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
17.28      17.63      16.93      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
22.78      23.27      16.93      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
25.83      25.46      25.30      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
28.23      28.87      25.30      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
47.68      47.72      45.65      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
49.25      49.45      45.65      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
38.73      38.18      37.89      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
33.68      33.99      29.16      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
36.22      29.50      29.16      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
36.12      35.69      29.16      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
219.81     220.21     215.97     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
224.03     222.27     215.97     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
81.46      84.07      77.95      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
81.46      83.07      77.95      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
136.14     136.12     128.61     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
136.57     136.30     128.61     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
805.54     805.23     795.82     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
768.87     766.00     727.12     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
29.47      29.39      28.49      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
54.45      54.76      53.45      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
60.84      61.07      53.45      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
57.22      57.22      56.14      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
218.22     224.50     215.54     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
199.53     203.24     179.85     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
53.06      54.61      51.82      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
148.82     149.62     146.73     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
148.91     148.99     146.59     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
175.33     150.60     146.59     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
194.12     201.48     146.59     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
133.27     132.90     132.54     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
135.27     135.12     132.54     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
127.49     137.43     113.82     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
129.18     125.95     113.82     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
125.82     114.44     113.82     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
127.63     124.81     113.82     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
129.24     134.50     113.82     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
126.64     125.09     110.45     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

### Jetson Nano Orin

Specs: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
- CPU: 6-core Arm® Cortex®-A78AE v8.2 64-bit CPU, 1.5MB L2 + 4MB L3
- GPU: 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores, max freq 625MHz

CPU:

```
$ python3 benchmark.py --all
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
2.59       2.62       2.50       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
2.98       2.97       2.50       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
20.05      24.76      19.75      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
31.84      32.72      19.75      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
9.15       9.22       9.04       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
14.33      15.35      9.04       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
15.00      15.17      14.80      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
18.37      18.63      14.80      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
24.86      25.09      24.12      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
30.17      34.51      24.12      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
18.47      18.55      18.23      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
17.08      17.30      15.80      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
21.26      15.89      15.80      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
23.19      24.15      15.80      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
102.30     101.90     101.44     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
142.33     146.24     101.44     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
39.91      39.01      38.46      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
51.35      50.70      38.46      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
125.31     126.50     121.92     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
132.95     133.67     121.92     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
400.91     430.48     384.87     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
476.63     509.48     384.87     [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
19.16      19.91      18.04      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
27.73      26.93      26.72      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
35.16      41.14      26.72      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
33.05      33.18      32.67      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
93.58      94.02      92.36      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
119.80     153.20     92.36      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
31.51      32.19      30.69      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
3.53       3.53       3.51       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
78.10      77.77      77.17      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
78.03      78.38      77.17      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
99.09      79.42      77.17      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
112.82     116.06     77.17      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
142.97     142.84     135.56     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
144.53     148.52     135.56     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
134.47     146.62     112.91     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
136.37     131.39     112.91     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
132.08     117.15     109.24     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
135.17     130.23     109.24     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
138.38     143.25     109.24     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
137.08     134.22     109.24     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

GPU (CUDA-FP32):

```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 1
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA
mean       median     min        input size   model
5.23       5.27       5.17       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
7.59       7.62       7.55       [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
8.48       8.46       8.37       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
12.29      13.04      11.11      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
12.91      13.28      12.79      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
8.41       8.42       8.35       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
9.36       9.43       8.35       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
32.58      32.71      31.11      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
16.33      16.08      16.04      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
24.46      24.35      24.01      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
103.28     103.41     102.37     [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
19.75      19.78      19.10      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
10.84      10.76      10.75      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
14.50      14.50      14.36      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
23.53      23.36      23.16      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
26.54      27.22      25.99      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
27.49      27.80      26.97      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
27.53      27.75      26.95      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
15.66      16.30      15.41      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
15.91      15.80      15.41      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
13.58      16.70      9.48       [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

GPU (CUDA-FP16):

```
$ python3 benchmark.py --all --fp32 --cfg_exclude wechat --cfg_overwrite_backend_target 2
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_CUDA
target=cv.dnn.DNN_TARGET_CUDA_FP16
mean       median     min        input size   model
5.00       5.04       4.92       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
5.09       5.08       5.05       [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
6.81       6.86       6.66       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
9.19       10.18      9.06       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
16.20      16.62      15.93      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
6.84       6.82       6.80       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
7.46       7.87       6.80       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
14.18      14.16      14.03      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
13.35      13.10      13.04      [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
19.94      19.95      19.50      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
72.25      72.91      70.99      [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
22.37      22.44      21.60      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
8.92       8.92       8.84       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
11.11      11.13      10.98      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
13.22      13.23      13.12      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
26.79      27.04      26.24      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
19.71      19.75      19.47      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
19.76      19.93      19.47      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
16.30      15.88      15.80      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
16.36      16.51      15.80      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
13.64      16.27      8.90       [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
```

### Atlas 200I DK A2

Specs: https://www.hiascend.com/hardware/developer-kit-a2 (cn)
- CPU: 4 core * 1.0 GHz
- NPU: Ascend 310B, 8 TOPS INT8, 4 TFLOPS FP16 (Benchmark results are coming later)

CPU:
<!-- config wechat is excluded due to it needs building with opencv_contrib -->
```
$ python3 benchmark.py --all --cfg_exclude wechat
Benchmarking ...
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
6.67       6.80       5.17       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
8.70       9.22       5.17       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
78.90      81.48      74.18      [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
113.79     115.49     74.18      [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
36.94      38.64      33.23      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
55.14      60.34      33.23      [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
56.00      55.56      51.99      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
71.09      72.20      51.99      [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
78.01      80.36      73.97      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
111.56     113.84     73.97      [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
70.20      68.69      65.12      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
61.72      63.39      48.28      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
80.12      54.37      48.28      [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
87.42      96.71      48.28      [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
417.31     417.30     406.17     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
597.15     619.24     406.17     [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
155.73     153.40     145.10     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
200.41     200.24     145.10     [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
253.05     252.73     245.91     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
274.44     269.76     245.91     [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
1407.75    1416.44    1357.23    [640, 640]   YoloX with ['object_detection_yolox_2022nov.onnx']
1716.25    1709.35    1357.23    [640, 640]   YoloX with ['object_detection_yolox_2022nov_int8.onnx']
37.02      37.66      32.50      [1280, 720]  VitTrack with ['object_tracking_vittrack_2023sep.onnx']
92.56      97.78      87.87      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
119.29     123.56     87.87      [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
90.13      90.75      87.78      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
285.75     284.54     278.06     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
389.02     405.12     278.06     [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
83.16      85.91      77.83      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
219.28     220.74     214.53     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
217.18     227.44     207.15     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
319.73     210.22     207.15     [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
396.47     399.45     207.15     [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
165.34     172.10     156.36     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
169.22     174.21     156.36     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
158.82     172.23     135.52     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
159.39     156.42     135.52     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
155.87     146.82     135.52     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
163.43     152.16     135.52     [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
173.46     162.85     135.52     [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
175.28     145.22     135.52     [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```
