# OpenCV Model Zoo报告 - 模型、板卡和性能基准结果分析

[![benchmark_table](assets/benchmark_table_4.9.0.png)](benchmark_table)

[OpenCV Model Zoo](https://github.com/opencv/opencv_zoo)项目于2021年9月启动。从那时起，我们已收集了43个模型权重，涵盖19个任务，并添加了13种硬件设置，涵盖不同的CPU架构（x86-64、ARM和RISC-V）以及不同的计算单元（CPU、GPU和NPU）。所有这些模型和硬件都经过我们的全面测试，并保证与OpenCV的最新版本（目前是4.9.0）兼容，如我们的基准表所示。

## Models

截至此版本发布，我们在opencv_zoo中共有43个模型权重，涵盖了总共19个任务。这些模型是考虑到许可证的，这意味着基本上您可以为任何目的使用opencv_zoo中的所有模型，甚至用于商业用途。它们主要来自以下4个来源：

- OpenCV中国团队。YuNet人脸检测模型由我们团队的一名成员开发和维护。
- OpenCV Area Chair。这是由OpenCV基金会启动的一个项目，详情可以在[这里](https://opencv.org/opencv-area-chairs/)找到。人脸识别的SFace模型和面部表情识别的FER模型是由Area Chair邓教授贡献的。
- 与OpenCV的合作。人体分割模型来自百度PaddlePaddle，修改后的YuNet用于车牌检测来自[watrix.ai](watrix.ai)。
- OpenCV社区。从2022年开始，我们在Google Summer of Code（GSoC）计划中有关于模型贡献的项目想法。GSoC学生已成功贡献了6个模型，涵盖了目标检测、目标跟踪和光流估计等任务。

我们欢迎您的贡献！

此外，我们为每个模型提供了在最新版本的OpenCV中可立即使用的Python和C++演示。我们还提供了[可视化样例](https://github.com/opencv/opencv_zoo?tab=readme-ov-file#some-examples)，以便开发者们更好地了解任务和输出的类型。

## Boards

opencv_zoo中有13种硬件设置，其中之一是搭载Intel i7-12700K的 PC，其他都是单板计算机（SBC）。它们按CPU架构分类如下：

<!-- TODO: add price -->

x86-64:

- Intel Core i7-12700K：8 P核（3.60GHz，4.90GHz turbo），4 E核（2.70GHz，3.80GHz turbo），20线程。

ARM:

| 板卡 | SoC 型号 | CPU 型号 | GPU 型号 | NPU 性能（Int8） |
| ----- | --- | --- | --- | --- |
| Khadas VIM3 | Amlogic A311D | 2.20GHz 四核 Cortex-A73 + 1.80GHz 双核 Cortex-A53 | ARM G52 | 5 TOPS |
| Khadas VIM4 | Amlogic A311D2 | 2.2GHz 四核 ARM Cortex-A73 + 2.0GHz 四核 Cortex-A53 | Mali G52MP8(8EE) 800Mhz  | 3.2 TOPS |
| Khadas Edge 2 | Rockchip RK3588S | 2.25GHz 四核 Cortex-A76 + 1.80GHz 四核 Cortex-A55 | 1GHz ARM Mali-G610 | 6 TOPS |
| Raspberry Pi 4B | Broadcom BCM2711 | 1.5GHz 四核 Cortex-A72 | *未知* | *无* |
| Horizon Sunrise X3 PI | Sunrise X3 | 1.2GHz 四核 Cortex-A53 | *未知* | 5 TOPS，双核伯努利架构|
| MAIX-III AXera-Pi | AXera AX620A | 四核 Cortex-A7 | *未知* | 3.6 TOPS |
| Toybrick RV1126 | Rockchip RV1126 | 四核 Cortex-A7 | *未知* | 2.0 TOPS |
| NVIDIA Jetson Nano B01 | *未知* | 1.43GHz 四核 Cortex-A57 | 128 核 NVIDIA Maxwell | *无* |
| NVIDIA Jetson Nano Orin | *未知* | 6 核 Cortex®-A78AE | 1024 核 NVIDIA Ampere | *无* |
| Atlas 200 DK | *未知* | *未知* | *未知* | 22 TOPS，Ascend 310 |
| Atlas 200I DK A2 | *未知* | 1.0GHz 四核 | *未知* | 8 TOPS，Ascend 310B |

RISC-V:

| 板卡 | SoC 型号 | CPU 型号 | GPU 型号 |
| ----- | --------- | --------- | --------- |
| StarFive VisionFive 2 | StarFive JH7110 | 1.5GHz 四核 RISC-V 64 位 | 600MHz IMG BXE-4-32 MC1 |
| Allwinner Nezha D1 | Allwinner D1 | 1.0GHz 单核 RISC-V 64 位，RVV-0.7.1 | *未知* |

我们的目标是在边缘设备上进行高效计算！在过去的几年中，我们（OpenCV）中国团队，已经在dnn模块针对ARM架构优化的方面付出了大量努力，特别关注卷积神经网络的卷积内核优化和Vision Transformers的GEMM内核优化。更值得一提的是，我们为dnn模块引入了NPU支持，支持Khadas VIM3、Atlas 200 DK 和Atlas 200I DK A2上的 NPU。在 NPU 上运行模型可以帮助将计算负载从CPU分配到NPU，甚至可以达到更快的推理速度（例如，在 Atlas 200 DK 上 Ascend 310 的测试结果）。
