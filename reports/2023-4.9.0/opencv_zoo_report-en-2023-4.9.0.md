# OpenCV Model Zoo Report - Models, Boards and Benchmark Result Analysis

<!-- ![benchmark_table_4.9.0](assets/benchmark_table.png) -->
[![benchmark_table](assets/benchmark_table_4.9.0.png)](benchmark_table)

[OpenCV Model Zoo](https://github.com/opencv/opencv_zoo) was started back in September, 2021. Since then, we have collected 43 model weights covering 19 tasks and added 13 hardware setups covering different CPU architectures (x86-64, ARM and RISC-V) and different computing units (CPU, GPU and NPU). All these models and hardware are fully tested by us and guaranteed to work with latest release of OpenCV (currently 4.9.0) as our benchmark table shown. 

## Models

As of this release, we have 43 model weights covering 19 tasks in total in the zoo. These models are collected with licenses in mind, meaning you can bascially use all the models in the zoo for whatever purposes you want, even for commercial purpose. They are collected from mainly 4 sources:

- OpenCV China team. The YuNet model for face detection is developed and maintained by one of our team members.
- OpenCV Area Chair. This is a program started by OpenCV Foundation, details can be found [here](https://opencv.org/opencv-area-chairs/). The SFace model for face recognition and FER model for facial expression recognition are contributed from one of the Area Chairs Prof. Deng.
- Cooperation with OpenCV. The HumanSeg model for human segmentation is from Baidu PaddlePaddle, and the modified YuNet for license plate detection is from [watrix.ai](watrix.ai).
- Community. Started from 2022, we have project ideas for model contribution in the Google Summer of Code (GSoC) program. GSoC students have successfully contributed 6 models covering tasks such as object detection, object tracking and optical flow estimation.

We welcome your contribution!

Besides, demos in Python and C++, which work out-of-the-box with latest OpenCV, are also provided for each model. We also provide [visual examples](https://github.com/opencv/opencv_zoo?tab=readme-ov-file#some-examples) so that people can better understand what the task is and what kind of the output is.

## Boards

There are 13 hardware setups in the zoo, one of them is a PC with Intel i7-12700K, and the others are single board computers (SBC). They are categorized by CPU architecture as follows:

<!-- TODO: add price -->

x86-64:

- Intel Core i7-12700K: 8 P-core (3.60GHz, 4.90GHz turbo), 4 E-core (2.70GHz, 3.80GHz turbo), 20 threads.


ARM:

| Board | SoC model | CPU model | GPU model | NPU Performance (Int8) |
| ----- | --- | --- | --- | --- |
| Khadas VIM3 | Amlogic A311D | 2.20GHz Quad-core Cortex-A73 + 1.80GHz Dual-core Cortex-A53 | ARM G52 | 5 TOPS |
| Khadas VIM4 | Amlogic A311D2 | 2.2GHz Quad-core ARM Cortex-A73 + 2.0GHz Quad-core Cortex-A53 | Mali G52MP8(8EE) 800Mhz  | 3.2 TOPS |
| Khadas Edge 2 | Rockchip RK3588S | 2.25GHz Quad-core Cortex-A76 + 1.80GHz Quad-core Cortex-A55 | 1GHz ARM Mali-G610 | 6 TOPS |
| Raspberry Pi 4B | Broadcom BCM2711 | 1.5GHz Quad-core Cortex-A72 | *Unknown* | *No* |
| Horizon Sunrise X3 PI | Sunrise X3 | 1.2GHz Quad-core Cortex-A53 | *Unkown* | 5 TOPS, Dual-core Bernoulli Arch|
| MAIX-III AXera-Pi | AXera AX620A | Quad-core Cortex-A7 | *Unknown* | 3.6 TOPS |
| Toybrick RV1126 | Rockchip RV1126 | Quad-core Cortex-A7 | *Unknown* | 2.0 TOPS |
| NVIDIA Jetson Nano B01 | *Unknown* | 1.43GHz Quad-core Cortex-A57 | 128-core NVIDIA Maxwell | *No* |
| NVIDIA Jetson Nano Orin | *Unknown* | 6-core Cortex®-A78AE | 1024-core NVIDIA Ampere | *No* |
| Atlas 200 DK | *Unknown* | *Unknown* | *Unknown* | 22 TOPS, Ascend 310 |
| Atlas 200I DK A2 | *Unknown* | 1.0GHz Quad-core | *Unknown* | 8 TOPS, Ascend 310B |


RISC-V:

| Board | SoC model | CPU model | GPU model |
| ----- | --------- | --------- | --------- |
| StarFive VisionFive 2 | StarFive JH7110 | 1.5GHz Quad-core RISC-V 64-bit | 600MHz IMG BXE-4-32 MC1 |
| Allwinner Nezha D1 | Allwinner D1 | 1.0GHz single-core RISC-V 64-bit, RVV-0.7.1 | *Unknown* |

We are targetting on efficient computing on edge devices! In the past few years, we, the OpenCV China team, have spent most of our effort in optimizing dnn module for ARM architecture, focusing especially on convolution kernel optimization for ConvNets and GEMM kernel optimization for Vision Transformers. What's even more worth mentioning is that we introduce NPU support for the dnn module, supporing the NPU in Khadas VIM3, Atlas 200 DK and Atlas 200I DK A2. Running the model on NPU can help distribute computing loads from CPU to NPU and even reaching a faster inference speed (see benchmark results on Ascend 310 on Atlas 200 DK for example).
