# OpenCV Zoo and Benchmark

A zoo for models tuned for OpenCV DNN with benchmarks on different platforms.

Guidelines:

- Install latest `opencv-python`:
  ```shell
  python3 -m pip install opencv-python
  # Or upgrade to latest version
  python3 -m pip install --upgrade opencv-python
  ```
- Clone this repo to download all models and demo scripts:
  ```shell
  # Install git-lfs from https://git-lfs.github.com/
  git clone https://github.com/opencv/opencv_zoo && cd opencv_zoo
  git lfs install
  git lfs pull
  ```
- To run benchmarks on your hardware settings, please refer to [benchmark/README](./benchmark/README.md).

## Models & Benchmark Results

| Model                                                   | Task                          | Input Size | CPU-INTEL (ms) | CPU-RPI (ms) | GPU-JETSON (ms) | NPU-KV3 (ms) | NPU-Ascend310 (ms) | CPU-D1 (ms) |
| ------------------------------------------------------- | ----------------------------- | ---------- | -------------- | ------------ | --------------- | ------------ | ------------------ | ----------- |
| [YuNet](./models/face_detection_yunet)                  | Face Detection                | 160x120    | 0.72           | 5.43         | 12.18           | 4.04         | 2.24               | 86.69       |
| [SFace](./models/face_recognition_sface)                | Face Recognition              | 112x112    | 6.04           | 78.83        | 24.88           | 46.25        | 2.66               | ---         |
| [FER](./models/facial_expression_recognition/)          | Facial Expression Recognition | 112x112    | 3.16           | 32.53        | 31.07           | 29.80        | 2.19               | ---         |
| [LPD-YuNet](./models/license_plate_detection_yunet/)    | License Plate Detection       | 320x240    | 8.63           | 167.70       | 56.12           | 29.53        | 7.63               | ---         |
| [YOLOX](./models/object_detection_yolox/)               | Object Detection              | 640x640    | 141.20         | 1805.87      | 388.95          | 420.98       | 28.59              | ---         |
| [NanoDet](./models/object_detection_nanodet/)           | Object Detection              | 416x416    | 66.03          | 225.10       | 64.94           | 116.64       | 20.62              | ---         |
| [DB-IC15](./models/text_detection_db) (EN)              | Text Detection                | 640x480    | 71.03          | 1862.75      | 208.41          | ---          | 17.15              | ---         |
| [DB-TD500](./models/text_detection_db) (EN&CN)          | Text Detection                | 640x480    | 72.31          | 1878.45      | 210.51          | ---          | 17.95              | ---         |
| [CRNN-EN](./models/text_recognition_crnn)               | Text Recognition              | 100x32     | 20.16          | 278.11       | 196.15          | 125.30       | ---                | ---         |
| [CRNN-CN](./models/text_recognition_crnn)               | Text Recognition              | 100x32     | 23.07          | 297.48       | 239.76          | 166.79       | ---                | ---         |
| [PP-ResNet](./models/image_classification_ppresnet)     | Image Classification          | 224x224    | 34.71          | 463.93       | 98.64           | 75.45        | 6.99               | ---         |
| [MobileNet-V1](./models/image_classification_mobilenet) | Image Classification          | 224x224    | 5.90           | 72.33        | 33.18           | 145.66\*     | 5.15               | ---         |
| [MobileNet-V2](./models/image_classification_mobilenet) | Image Classification          | 224x224    | 5.97           | 66.56        | 31.92           | 146.31\*     | 5.41               | ---         |
| [PP-HumanSeg](./models/human_segmentation_pphumanseg)   | Human Segmentation            | 192x192    | 8.81           | 73.13        | 67.97           | 74.77        | 6.94               | ---         |
| [WeChatQRCode](./models/qrcode_wechatqrcode)            | QR Code Detection and Parsing | 100x100    | 1.29           | 5.71         | ---             | ---          | ---                | ---         |
| [DaSiamRPN](./models/object_tracking_dasiamrpn)         | Object Tracking               | 1280x720   | 29.05          | 712.94       | 76.82           | ---          | ---                | ---         |
| [YoutuReID](./models/person_reid_youtureid)             | Person Re-Identification      | 128x256    | 30.39          | 625.56       | 90.07           | 44.61        | 5.58               | ---         |
| [MP-PalmDet](./models/palm_detection_mediapipe)         | Palm Detection                | 192x192    | 6.29           | 86.83        | 83.20           | 33.81        | 5.17               | ---         |
| [MP-HandPose](./models/handpose_estimation_mediapipe)   | Hand Pose Estimation          | 224x224    | 4.68           | 43.57        | 40.10           | 19.47        | 6.27               | ---         |
| [MP-PersonDet](./models/person_detection_mediapipe)     | Person Detection              | 224x224    | 13.88          | 98.52        | 56.69           | ---          | 16.45              | ---         |

\*: Models are quantized in per-channel mode, which run slower than per-tensor quantized models on NPU.

Hardware Setup:

- `CPU-INTEL`: [Intel Core i7-12700K](https://www.intel.com/content/www/us/en/products/sku/134594/intel-core-i712700k-processor-25m-cache-up-to-5-00-ghz/specifications.html), 8 Performance-cores (3.60 GHz, turbo up to 4.90 GHz), 4 Efficient-cores (2.70 GHz, turbo up to 3.80 GHz), 20 threads.
- `CPU-RPI`: [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/), Broadcom BCM2711, Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5 GHz.
- `GPU-JETSON`: [NVIDIA Jetson Nano B01](https://developer.nvidia.com/embedded/jetson-nano-developer-kit), 128-core NVIDIA Maxwell GPU.
- `NPU-KV3`: [Khadas VIM3](https://www.khadas.com/vim3), 5TOPS Performance. Benchmarks are done using **quantized** models. You will need to compile OpenCV with TIM-VX following [this guide](https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU) to run benchmarks. The test results use the `per-tensor` quantization model by default.
- `NPU-Ascend310`: [Ascend 310](https://e.huawei.com/uk/products/cloud-computing-dc/atlas/atlas-200), 22 TOPS @ INT8. Benchmarks are done on [Atlas 200 DK AI Developer Kit](https://e.huawei.com/in/products/cloud-computing-dc/atlas/atlas-200). Get the latest OpenCV source code and build following [this guide](https://github.com/opencv/opencv/wiki/Huawei-CANN-Backend) to enable CANN backend.
- `CPU-D1`: [Allwinner D1](https://d1.docs.aw-ol.com/en), [Xuantie C906 CPU](https://www.t-head.cn/product/C906?spm=a2ouz.12986968.0.0.7bfc1384auGNPZ) (RISC-V, RVV 0.7.1) @ 1.0 GHz, 1 core. YuNet is supported for now. Visit [here](https://github.com/fengyuentau/opencv_zoo_cpp) for more details.

***Important Notes***:

- The data under each column of hardware setups on the above table represents the elapsed time of an inference (preprocess, forward and postprocess).
- The time data is the mean of 10 runs after some warmup runs. Different metrics may be applied to some specific models.
- Batch size is 1 for all benchmark results.
- `---` represents the model is not availble to run on the device.
- View [benchmark/config](./benchmark/config) for more details on benchmarking different models.

## Some Examples

Some examples are listed below. You can find more in the directory of each model!

### Face Detection with [YuNet](./models/face_detection_yunet/)

![largest selfie](./models/face_detection_yunet/examples/largest_selfie.jpg)

### Facial Expression Recognition with [Progressive Teacher](./models/facial_expression_recognition/)

![fer demo](./models/facial_expression_recognition/examples/selfie.jpg)

### Human Segmentation with [PP-HumanSeg](./models/human_segmentation_pphumanseg/)

![messi](./models/human_segmentation_pphumanseg/examples/messi.jpg)

### License Plate Detection with [LPD_YuNet](./models/license_plate_detection_yunet/)

![license plate detection](./models/license_plate_detection_yunet/examples/lpd_yunet_demo.gif)

### Object Detection with [NanoDet](./models/object_detection_nanodet/) & [YOLOX](./models/object_detection_yolox/)

![nanodet demo](./models/object_detection_nanodet/samples/1_res.jpg)

![yolox demo](./models/object_detection_yolox/samples/3_res.jpg)

### Object Tracking with [DaSiamRPN](./models/object_tracking_dasiamrpn/)

![webcam demo](./models/object_tracking_dasiamrpn/examples/dasiamrpn_demo.gif)

### Palm Detection with [MP-PalmDet](./models/palm_detection_mediapipe/)

![palm det](./models/palm_detection_mediapipe/examples/mppalmdet_demo.gif)

### Hand Pose Estimation with [MP-HandPose](models/handpose_estimation_mediapipe/)

![handpose estimation](models/handpose_estimation_mediapipe/examples/mphandpose_demo.webp)

### QR Code Detection and Parsing with [WeChatQRCode](./models/qrcode_wechatqrcode/)

![qrcode](./models/qrcode_wechatqrcode/examples/wechat_qrcode_demo.gif)

### Chinese Text detection [DB](./models/text_detection_db/)

![mask](./models/text_detection_db/examples/mask.jpg)

### English Text detection [DB](./models/text_detection_db/)

![gsoc](./models/text_detection_db/examples/gsoc.jpg)

### Text Detection with [CRNN](./models/text_recognition_crnn/)

![crnn_demo](./models/text_recognition_crnn/examples/CRNNCTC.gif)

## License

OpenCV Zoo is licensed under the [Apache 2.0 license](./LICENSE). Please refer to licenses of different models.
