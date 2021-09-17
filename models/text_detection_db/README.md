# DB

Real-time Scene Text Detection with Differentiable Binarization

`text_detection_db.onnx` is trained on [TD500 dataset](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)), which can detect both English & Chinese instances. It is obtained from [here](https://docs.opencv.org/master/d4/d43/tutorial_dnn_text_spotting.html) and renamed from `DB_TD500_resnet18.onnx`.

## Demo

Run the following command to try the demo:
```shell
# detect on camera input
python demo.py
# detect on an image
python demo.py --input /path/to/image
```

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://arxiv.org/abs/1911.08947
- https://github.com/MhLiao/DB
- https://docs.opencv.org/master/d4/d43/tutorial_dnn_text_spotting.html