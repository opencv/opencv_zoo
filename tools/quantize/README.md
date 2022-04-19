# Quantization with ONNXRUNTIME and Neural Compressor

[ONNXRUNTIME](https://github.com/microsoft/onnxruntime) and [Neural Compressor](https://github.com/intel/neural-compressor) are used for quantization in the Zoo.

Install dependencies before trying quantization:
```shell
pip install -r requirements.txt
```

## Usage

Quantize all models in the Zoo:
```shell
python quantize.py
```

Quantize one of the models in the Zoo:
```shell
# python quantize.py <key_in_models>
python quantize.py yunet
```

Customizing quantization configs:
```python
# add model into `models` dict in quantize.py
models = dict(
    # ...
    model1=Quantize(model_path='/path/to/model1.onnx'
                    calibration_image_dir='/path/to/images',
                    transforms=Compose([''' transforms ''']), # transforms can be found in transforms.py
                    per_channel=False, # set False to quantize in per-tensor style
                    act_type='int8',   # available types: 'int8', 'uint8'
                    wt_type='int8'     # available types: 'int8', 'uint8'
    )
)
# quantize the added models
python quantize.py model1
```
