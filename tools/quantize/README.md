# Quantization with ONNXRUNTIME and Neural Compressor

[ONNXRUNTIME](https://github.com/microsoft/onnxruntime) and [Neural Compressor](https://github.com/intel/neural-compressor) are used for quantization in the Zoo.

Install dependencies before trying quantization:
```shell
pip install -r requirements.txt
```

## Usage

Quantize all models in the Zoo:
```shell
python quantize-ort.py
python quantize-inc.py
```

Quantize one of the models in the Zoo:
```shell
# python quantize.py <key_in_models>
python quantize-ort.py yunet
python quantize-inc.py mobilenetv1
```

Customizing quantization configs:
```python
# Quantize with ONNXRUNTIME
# 1. add your model into `models` dict in quantize-ort.py
models = dict(
    # ...
    model1=Quantize(model_path='/path/to/model1.onnx',
                    calibration_image_dir='/path/to/images',
                    transforms=Compose([''' transforms ''']), # transforms can be found in transforms.py
                    per_channel=False, # set False to quantize in per-tensor style
                    act_type='int8',   # available types: 'int8', 'uint8'
                    wt_type='int8'     # available types: 'int8', 'uint8'
    )
)
# 2. quantize your model
python quantize-ort.py model1


# Quantize with Intel Neural Compressor
# 1. add your model into `models` dict in quantize-inc.py
models = dict(
    # ...
    model1=Quantize(model_path='/path/to/model1.onnx',
                    config_path='/path/to/model1.yaml'),
)
# 2. prepare your YAML config model1.yaml (see configs in ./inc_configs)
# 3. quantize your model
python quantize-inc.py model1
```
