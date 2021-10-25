# OpenCV Zoo

A zoo for models tuned for OpenCV DNN with benchmarks on different platforms.

Guidelines:
- To clone this repo, please install [git-lfs](https://git-lfs.github.com/), run `git lfs install` and use `git lfs clone https://github.com/opencv/opencv_zoo`.
- To run benchmark on your hardware settings, please refer to [benchmark/README](./benchmark/README.md).

## Models & Benchmarks

Hardware Setup:
- `CPU x86_64`: INTEL CPU i7-5930K @ 3.50GHz, 6 cores, 12 threads.
- `CPU ARM`: Raspberry 4B, BCM2711B0 @ 1.5GHz (Cortex A-72), 4 cores, 4 threads.
- `GPU CUDA`: NVIDIA Jetson Nano B01, 128-core Maxwell, Quad-core ARM A57 @ 1.43 GHz.

***Important Notes***:
- The time data that shown on the following table presents the time elapsed from preprocess (resize is excluded), to a forward pass of a network, and postprocess to get final results.
- The time data that shown on the following table is the median of 10 runs. Different metrics may be applied to some specific models.
- View [benchmark/config](./benchmark/config) for more details on benchmarking different models.

| Model | Input Size | CPU x86_64 (ms) | CPU ARM (ms) | GPU CUDA (ms) |
|-------|------------|-----------------|--------------|---------------|
| [YuNet](./models/face_detection_yunet)   | 160x120 | 1.45   | 6.22    | TBD |
| [DB](./models/text_detection_db)         | 640x480 | 137.38 | 2780.78 | TBD |
| [CRNN](./models/text_recognition_crnn)   | 100x32  | 50.21  | 234.32  | TBD |
| [SFace](./models/face_recognition_sface) | 112x112 | 8.65 | 99.20 | TBD |
| [PP-ResNet](./models/image_classification_ppresnet) | 224x224 | 56.05 | 602.58 TBD |
| [PP-HumanSeg](./models/human_segmentation_pphumanseg) | 192x192 | 19.92 | 105.32 | TBD |

## License

OpenCV Zoo is licensed under the [Apache 2.0 license](./LICENSE). Please refer to licenses of different models.
