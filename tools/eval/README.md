# Accuracy evaluation of models in OpenCV Zoo

Make sure you have the following packages installed:

```shell
pip install tqdm
```

Generally speaking, evaluation can be done with the following command:

```shell
python eval.py -m model_name -d dataset_name -dr dataset_root_dir
```

Supported datasets:
- [ImageNet](./datasets/imagenet.py)
- [ICDAR](./datasets/icdar.py)

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

## ICDAR2003

### Prepare data

Please visit http://iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions to download the ICDAR2003 dataset and the labels. 

### Evaluation

Run evaluation with the following command:

```shell
python eval.py -m mobilenet -d imagenet -dr /path/to/icdar
```