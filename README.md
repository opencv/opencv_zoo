# OpenCV Zoo

A zoo for models tuned for OpenCV DNN with benchmarks on different platforms.

Guidelines:
- To clone this repo, please install [git-lfs](https://git-lfs.github.com/), run `git lfs install` and use `git lfs clone https://github.com/opencv/opencv_zoo`.
- To run benchmark on your hardware settings, please refer to [benchmark/README](./benchmark/README.md).
- Understand model filename: `<topic>_<model_name>_<dataset>_<arch>_<upload_time>`
    - `<topic>`: research topics, such as `face detection` etc.
    - `<model_name>`: exact model names.
    - `<dataset>`: (Optional) the dataset that the model is trained with.
    - `<arch>`: (Optional) the backbone architecture of the model.
    - `<upload_time>`: the time when the model is uploaded, meaning the latest version of this model unless specified.

## Models & Benchmarks

| Model | Input Size | CPU x86_64 (ms) | CPU ARM (ms) | GPU CUDA (ms) |
|-------|------------|-----------------|--------------|---------------|
| [YuNet](./models/face_detection_yunet)   | 160x120 | 1.45   | 6.22    | 12.18 |
| [DB-IC15](./models/text_detection_db)    | 640x480 | 142.91 | 2835.91 | 208.41 |
| [DB-TD500](./models/text_detection_db)   | 640x480 | 142.91 | 2841.71 | 210.51 |
| [CRNN](./models/text_recognition_crnn)   | 100x32  | 50.21  | 234.32  | 196.15 |
| [SFace](./models/face_recognition_sface) | 112x112 | 8.65 | 99.20 | 24.88 |
| [PP-ResNet](./models/image_classification_ppresnet) | 224x224 | 56.05 | 602.58 | 98.64 |
| [PP-HumanSeg](./models/human_segmentation_pphumanseg) | 192x192 | 19.92 | 105.32 | 67.97 |
| [WeChatQRCode](./models/qrcode_wechatqrcode) | 100x100 | 7.04 | 37.68 | --- |
| [DaSiamRPN](./models/object_tracking_dasiamrpn) | 1280x720 | 36.15 | 705.48 | 76.82 |

Hardware Setup:
- `CPU x86_64`: INTEL CPU i7-5930K @ 3.50GHz, 6 cores, 12 threads.
- `CPU ARM`: Raspberry 4B, BCM2711B0 @ 1.5GHz (Cortex A-72), 4 cores, 4 threads.
- `GPU CUDA`: NVIDIA Jetson Nano B01, 128-core Maxwell, Quad-core ARM A57 @ 1.43 GHz.

***Important Notes***:
- The time data that shown on the following table presents the time elapsed from preprocess (resize is excluded), to a forward pass of a network, and postprocess to get final results.
- The time data that shown on the following table is the median of 10 runs. Different metrics may be applied to some specific models.
- Batch size is 1 for all benchmark results.
- View [benchmark/config](./benchmark/config) for more details on benchmarking different models.
- `---` means this model is not availble to run on the device.

## License

OpenCV Zoo is licensed under the [Apache 2.0 license](./LICENSE). Please refer to licenses of different models.
