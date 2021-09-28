# OpenCV Zoo Benchmark

Benchmarking different models in the zoo.

Data for benchmarking will be downloaded and loaded in [data](./data) based on given config.

Time is measured from data preprocess (resize is excluded), to a forward pass of a network, and postprocess to get final results. The final time data presented is averaged from a 100-time run.

## Preparation

1. Install `python >= 3.6`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download data for benchmarking.
    1. Download all data: `python download_data.py`
    2. Download one or more specified data: `python download_data.py face text`. Available names can be found in `download_data.py`.
    3. If download fails, you can download all data from https://pan.baidu.com/s/18sV8D4vXUb2xC9EG45k7bg (code: pvrw). Please place and extract data packages under [./data](./data).

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