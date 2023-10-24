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

![](benchmark/color_table.svg?raw=true)

Hardware Setup:

- [Intel Core i7-12700K](https://www.intel.com/content/www/us/en/products/sku/134594/intel-core-i712700k-processor-25m-cache-up-to-5-00-ghz/specifications.html): 8 Performance-cores (3.60 GHz, turbo up to 4.90 GHz), 4 Efficient-cores (2.70 GHz, turbo up to 3.80 GHz), 20 threads.
- [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/): Broadcom BCM2711 SoC with a Quad core Cortex-A72 (ARM v8) 64-bit @ 1.5 GHz.
- [Toybrick RV1126](https://t.rock-chips.com/en/portal.php?mod=view&aid=26): Rockchip RV1126 SoC with a quard-core ARM Cortex-A7 CPU and a 2.0 TOPs NPU.
- [Khadas Edge 2](https://www.khadas.com/edge2): Rockchip RK3588S SoC with a CPU of 2.25 GHz Quad Core ARM Cortex-A76 + 1.8 GHz Quad Core Cortex-A55, and a 6 TOPS NPU.
- [Horizon Sunrise X3](https://developer.horizon.ai/sunrise): an SoC from Horizon Robotics with a quad-core ARM Cortex-A53 1.2 GHz CPU and a 5 TOPS BPU (a.k.a NPU).
- [MAIX-III AXera-Pi](https://wiki.sipeed.com/hardware/en/maixIII/ax-pi/axpi.html#Hardware): Axera AX620A SoC with a quad-core ARM Cortex-A7 CPU and a 3.6 TOPS @ int8 NPU.
- [StarFive VisionFive 2](https://doc-en.rvspace.org/VisionFive2/Product_Brief/VisionFive_2/specification_pb.html): `StarFive JH7110` SoC with a RISC-V quad-core CPU, which can turbo up to 1.5GHz, and an GPU of model `IMG BXE-4-32 MC1` from Imagination, which has a work freq up to 600MHz.
- [NVIDIA Jetson Nano B01](https://developer.nvidia.com/embedded/jetson-nano-developer-kit): a Quad-core ARM A57 @ 1.43 GHz CPU, and a 128-core NVIDIA Maxwell GPU.
- [Khadas VIM3](https://www.khadas.com/vim3): Amlogic A311D SoC with a 2.2GHz Quad core ARM Cortex-A73 + 1.8GHz dual core Cortex-A53 ARM CPU, and a 5 TOPS NPU. Benchmarks are done using **per-tensor quantized** models. Follow [this guide](https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU) to build OpenCV with TIM-VX backend enabled.
- [Atlas 200 DK](https://e.huawei.com/en/products/computing/ascend/atlas-200): Ascend 310 NPU with 22 TOPS @ INT8. Follow [this guide](https://github.com/opencv/opencv/wiki/Huawei-CANN-Backend) to build OpenCV with CANN backend enabled.
- [Allwinner Nezha D1](https://d1.docs.aw-ol.com/en): Allwinner D1 SoC with a 1.0 GHz single-core RISC-V [Xuantie C906 CPU](https://www.t-head.cn/product/C906?spm=a2ouz.12986968.0.0.7bfc1384auGNPZ) with RVV 0.7.1 support. YuNet is tested for now. Visit [here](https://github.com/fengyuentau/opencv_zoo_cpp) for more details.

***Important Notes***:

- The data under each column of hardware setups on the above table represents the elapsed time of an inference (preprocess, forward and postprocess).
- The time data is the mean of 10 runs after some warmup runs. Different metrics may be applied to some specific models.
- Batch size is 1 for all benchmark results.
- `---` represents the model is not availble to run on the device.
- View [benchmark/config](./benchmark/config) for more details on benchmarking different models.

## Some Examples

Some examples are listed below. You can find more in the directory of each model!

### Face Detection with [YuNet](./models/face_detection_yunet/)

![largest selfie](./models/face_detection_yunet/example_outputs/largest_selfie.jpg)

### Facial Expression Recognition with [Progressive Teacher](./models/facial_expression_recognition/)

![fer demo](./models/facial_expression_recognition/example_outputs/selfie.jpg)

### Human Segmentation with [PP-HumanSeg](./models/human_segmentation_pphumanseg/)

![messi](./models/human_segmentation_pphumanseg/example_outputs/messi.jpg)

### License Plate Detection with [LPD_YuNet](./models/license_plate_detection_yunet/)

![license plate detection](./models/license_plate_detection_yunet/example_outputs/lpd_yunet_demo.gif)

### Object Detection with [NanoDet](./models/object_detection_nanodet/) & [YOLOX](./models/object_detection_yolox/)

![nanodet demo](./models/object_detection_nanodet/example_outputs/1_res.jpg)

![yolox demo](./models/object_detection_yolox/example_outputs/3_res.jpg)

### Object Tracking with [VitTrack](./models/object_tracking_vittrack/)

![webcam demo](./models/object_tracking_vittrack/example_outputs/vittrack_demo.gif)

### Palm Detection with [MP-PalmDet](./models/palm_detection_mediapipe/)

![palm det](./models/palm_detection_mediapipe/example_outputs/mppalmdet_demo.gif)

### Hand Pose Estimation with [MP-HandPose](models/handpose_estimation_mediapipe/)

![handpose estimation](models/handpose_estimation_mediapipe/example_outputs/mphandpose_demo.webp)

### Person Detection with [MP-PersonDet](./models/person_detection_mediapipe)

![person det](./models/person_detection_mediapipe/example_outputs/mppersondet_demo.webp)

### Pose Estimation with [MP-Pose](models/pose_estimation_mediapipe)

![pose_estimation](models/pose_estimation_mediapipe/example_outputs/mpposeest_demo.webp)

### QR Code Detection and Parsing with [WeChatQRCode](./models/qrcode_wechatqrcode/)

![qrcode](./models/qrcode_wechatqrcode/example_outputs/wechat_qrcode_demo.gif)

### Chinese Text detection [DB](./models/text_detection_db/)

![mask](./models/text_detection_db/example_outputs/mask.jpg)

### English Text detection [DB](./models/text_detection_db/)

![gsoc](./models/text_detection_db/example_outputs/gsoc.jpg)

### Text Detection with [CRNN](./models/text_recognition_crnn/)

![crnn_demo](./models/text_recognition_crnn/example_outputs/CRNNCTC.gif)

## License

OpenCV Zoo is licensed under the [Apache 2.0 license](./LICENSE). Please refer to licenses of different models.
