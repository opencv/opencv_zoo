# OpenCV Zoo and Benchmark

A zoo for models tuned for OpenCV DNN with benchmarks on different platforms.

Guidelines:
- Clone this repo to download all models and demo scripts:
    ```shell
    # Install git-lfs from https://git-lfs.github.com/
    git clone https://github.com/opencv/opencv_zoo && cd opencv_zoo
    git lfs install
    git lfs pull
    ```
- To run benchmarks on your hardware settings, please refer to [benchmark/README](./benchmark/README.md).

## Models & Benchmark Results

| Model | Input Size | INTEL-CPU | RPI-CPU | JETSON-GPU | TB-CPU |
|-------|------------|-----------|---------|------------|--------|
| [YuNet](./models/face_detection_yunet)   | 160x120 | 1.45   | 6.22    | 12.18 | 30.45 |
| [DB-IC15](./models/text_detection_db)    | 640x480 | 142.91 | 2835.91 | 208.41 | 15631.20 |
| [DB-TD500](./models/text_detection_db)   | 640x480 | 142.91 | 2841.71 | 210.51 | 15611.51 |
| [CRNN-EN](./models/text_recognition_crnn)   | 100x32  | 50.21  | 234.32  | 196.15 | 1194.47 |
| [CRNN-CN](./models/text_recognition_crnn)   | 100x32  | 73.52  | 322.16  | 239.76 | 1335.90 |
| [SFace](./models/face_recognition_sface) | 112x112 | 8.65 | 99.20 | 24.88 | 558.46 |
| [PP-ResNet](./models/image_classification_ppresnet) | 224x224 | 56.05 | 602.58 | 98.64 | 3822.21 |
| [PP-HumanSeg](./models/human_segmentation_pphumanseg) | 192x192 | 19.92 | 105.32 | 67.97 | 578.53 |
| [WeChatQRCode](./models/qrcode_wechatqrcode) | 100x100 | 7.04 | 37.68 | --- | 164.47 |
| [DaSiamRPN](./models/object_tracking_dasiamrpn) | 1280x720 | 36.15 | 705.48 | 76.82 | 4470.03 |
| [YoutuReID](./models/person_reid_youtureid) | 128x256 | 35.81 | 521.98 | 90.07 | 3299.95 |

Hardware Setup:
- `INTEL-CPU`: [Intel Core i7-5930K](https://www.intel.com/content/www/us/en/products/sku/82931/intel-core-i75930k-processor-15m-cache-up-to-3-70-ghz/specifications.html) @ 3.50GHz, 6 cores, 12 threads.
- `RPI-CPU`: [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/), Broadcom BCM2711, Quad core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz.
- `JETSON-GPU`: [NVIDIA Jetson Nano B01](https://developer.nvidia.com/embedded/jetson-nano-developer-kit), 128-core NVIDIA Maxwell GPU.
- `TB-CPU`: [Toybrick RV1126Ds](https://t.rock-chips.com/en/wiki.php?filename=Board_Guide/TB-RV1126D#hash_2), Quard core ARM Cortex-A7 (ARM v7-A) 32-bit.

***Important Notes***:
- The data under each column of hardware setups on the above table represents the elapsed time of an inference (preprocess, forward and postprocess).
- The time data is the median of 10 runs after some warmup runs. Different metrics may be applied to some specific models.
- Batch size is 1 for all benchmark results.
- `---` represents the model is not availble to run on the device.
- View [benchmark/config](./benchmark/config) for more details on benchmarking different models.

## License

OpenCV Zoo is licensed under the [Apache 2.0 license](./LICENSE). Please refer to licenses of different models.
