# DB

Real-time Scene Text Detection with Differentiable Binarization

Note:

- Models source: [here](https://drive.google.com/drive/folders/1qzNCHfUJOS0NEUOIKn69eCtxdlNPpWbq).
- `IC15` in the filename means the model is trained on [IC15 dataset](https://rrc.cvc.uab.es/?ch=4&com=introduction), which can detect English text instances only.
- `TD500` in the filename means the model is trained on [TD500 dataset](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500)), which can detect both English & Chinese instances.
- Visit https://docs.opencv.org/master/d4/d43/tutorial_dnn_text_spotting.html for more information.

## Demo

Run the following command to try the demo:

```shell
# detect on camera input
python demo.py
# detect on an image
python demo.py --input /path/to/image

# get help regarding various parameters
python demo.py --help
```

### Example outputs

![mask](./examples/mask.jpg)

![gsoc](./examples/gsoc.jpg)

## License

All files in this directory are licensed under [Apache 2.0 License](./LICENSE).

## Reference

- https://arxiv.org/abs/1911.08947
- https://github.com/MhLiao/DB
- https://docs.opencv.org/master/d4/d43/tutorial_dnn_text_spotting.html
