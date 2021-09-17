# OpenCV Zoo Benchmark

Benchmarking different models in the zoo.

Data for benchmarking will be downloaded and loaded in [data](./data) based on given config.

Time is measured from data preprocess (resize is excluded), to a forward pass of a network, and postprocess to get final results. The final time data presented is averaged from a 100-time run.

## Preparation

1. Install `python >= 3.6`.
2. Install dependencies: `pip install -r requirements.txt`.

## Benchmarking

Run the following command to benchmark on a given config:

```shell
PYTHONPATH=.. python benchmark.py --cfg ./config/face_detection_yunet.yaml
```

If you are a Windows user and wants to run in CMD/PowerShell, use this command instead:
```shell
set PYTHONPATH=..
python benchmark.py --cfg ./config/face_detection_yunet.yaml
```
<!--
Omit `--cfg` if you want to benchmark all included models:
```shell
PYTHONPATH=.. python benchmark.py
```
-->