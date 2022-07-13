# Accuracy evaluation of models in OpenCV Zoo

Make sure you have the following packages installed:

```shell
pip install tqdm
pip install scikit-learn
```

Generally speaking, evaluation can be done with the following command:

```shell
python eval.py -m model_name -d dataset_name -dr dataset_root_dir
```

Supported datasets:
- [ImageNet](./datasets/imagenet.py)
- [LFW](#lfw)

## ImageNet

### Prepare data

Please visit https://image-net.org/ to download the ImageNet dataset and [the labels from caffe](http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz). Organize files as follow:

```shell
$ tree -L 2 /path/to/imagenet
.
├── caffe_ilsvrc12
│   ├── det_synset_words.txt
│   ├── imagenet.bet.pickle
│   ├── imagenet_mean.binaryproto
│   ├── synsets.txt
│   ├── synset_words.txt
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── caffe_ilsvrc12.tar.gz
├── ILSVRC
│   ├── Annotations
│   ├── Data
│   └── ImageSets
├── imagenet_object_localization_patched2019.tar.gz
├── LOC_sample_submission.csv
├── LOC_synset_mapping.txt
├── LOC_train_solution.csv
└── LOC_val_solution.csv
```

### Evaluation

Run evaluation with the following command:

```shell
python eval.py -m mobilenet -d imagenet -dr /path/to/imagenet
```

## LFW
The script is modified based on [evaluation of InsightFace](https://github.com/deepinsight/insightface/blob/f92bf1e48470fdd567e003f196f8ff70461f7a20/src/eval/lfw.py).

### Prepare data

Please visit http://vis-www.cs.umass.edu/lfw to download the LFW [all images](http://vis-www.cs.umass.edu/lfw/lfw.tgz)(needs to be decompressed) and [pairs.txt](http://vis-www.cs.umass.edu/lfw/pairs.txt)(needs to be placed in the `view2` folder). Organize files as follow:

```shell
$ tree -L 2 /path/to/lfw
.
├── lfw
│   ├── Aaron_Eckhart
│   ├── ...
│   └── Zydrunas_Ilgauskas
└── view2
    └── pairs.txt
```

### Evaluation

Run evaluation with the following command:

```shell
python eval.py -m sface -d lfw -dr /path/to/lfw
```