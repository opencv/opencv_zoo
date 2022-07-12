# Accuracy evaluation of models in OpenCV Zoo

Make sure you have the following packages installed:

```shell
pip install tqdm
pip install scipy
```

Generally speaking, evaluation can be done with the following command:

```shell
python eval.py -m model_name -d dataset_name -dr dataset_root_dir
```

Supported datasets:
- [ImageNet](##imagenet)
- [WIDERFace](##widerface)

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

## WIDERFace

The script is modified based on [WiderFace-Evaluation](https://github.com/wondervictor/WiderFace-Evaluation).

### Prepare data

Please visit http://shuoyang1213.me/WIDERFACE to download the WIDERFace dataset [Validation Images](https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip), [Face annotations](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip) and [eval_tools](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip). Organize files as follow:

```shell
$ tree -L 2 /path/to/widerface
.
├── eval_tools
│   ├── boxoverlap.m
│   ├── evaluation.m
│   ├── ground_truth
│   ├── nms.m
│   ├── norm_score.m
│   ├── plot
│   ├── read_pred.m
│   └── wider_eval.m
├── wider_face_split
│   ├── readme.txt
│   ├── wider_face_test_filelist.txt
│   ├── wider_face_test.mat
│   ├── wider_face_train_bbx_gt.txt
│   ├── wider_face_train.mat
│   ├── wider_face_val_bbx_gt.txt
│   └── wider_face_val.mat
└── WIDER_val
    └── images
```

### Evaluation

Run evaluation with the following command:

```shell
python eval.py -m yunet -d widerface -dr /path/to/widerface
```
