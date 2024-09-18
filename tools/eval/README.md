# Accuracy evaluation of models in OpenCV Zoo

Make sure you have the following packages installed:

```shell
pip install tqdm
pip install scikit-learn
pip install scipy==1.8.1
```

Generally speaking, evaluation can be done with the following command:

```shell
python eval.py -m model_name -d dataset_name -dr dataset_root_dir
```

Supported datasets:

- [ImageNet](#imagenet)
- [WIDERFace](#widerface)
- [LFW](#lfw)
- [ICDAR](#icdar2003)
- [IIIT5K](#iiit5k)
- [Mini Supervisely](#mini-supervisely)

## ImageNet

### Prepare data

Please visit https://image-net.org/ to download the ImageNet dataset (only need images in `ILSVRC/Data/CLS-LOC/val`) and [the labels from caffe](http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz). Organize files as follow:

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

## LFW

The script is modified based on [evaluation of InsightFace](https://github.com/deepinsight/insightface/blob/f92bf1e48470fdd567e003f196f8ff70461f7a20/src/eval/lfw.py).

This evaluation uses [YuNet](../../models/face_detection_yunet) as face detector. The structure of the face bounding boxes saved in [lfw_face_bboxes.npy](../eval/datasets/lfw_face_bboxes.npy) is shown below.
Each row represents the bounding box of the main face that will be used in each image.

```shell
[
  [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm],
  ...
  [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm]
]
```

`x1, y1, w, h` are the top-left coordinates, width and height of the face bounding box, `{x, y}_{re, le, nt, rcm, lcm}` stands for the coordinates of right eye, left eye, nose tip, the right corner and left corner of the mouth respectively. Data type of this numpy array is `np.float32`.


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

## ICDAR2003

### Prepare data

Please visit http://iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions to download the ICDAR2003 dataset and the labels. You have to download the Robust Word Recognition [TrialTrain Set](http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/word.zip) only.

```shell
$ tree -L 2 /path/to/icdar
.
├── word
│   ├── 1
│   │   ├── self
│   │   ├── ...
│   │   └── willcooks
│   ├── ...
│   └── 12
└── word.xml
    
```

### Evaluation

Run evaluation with the following command:

```shell
python eval.py -m crnn -d icdar -dr /path/to/icdar
```

### Example

```shell
download zip file from http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/word.zip
upzip file to /path/to/icdar
python eval.py -m crnn -d icdar -dr /path/to/icdar
```

## IIIT5K

### Prepare data

Please visit https://github.com/cv-small-snails/Text-Recognition-Material to download the IIIT5K dataset and the labels.

### Evaluation

All the datasets in the format of lmdb can be evaluated by this script.<br>
Run evaluation with the following command:

```shell
python eval.py -m crnn -d iiit5k -dr /path/to/iiit5k
```


## Mini Supervisely

### Prepare data
Please download the mini_supervisely data from [here](https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip) which includes the validation dataset and unzip it.

```shell
$ tree -L 2 /path/to/mini_supervisely
.
├── Annotations
│   ├── ache-adult-depression-expression-41253.png
│   ├── ...
├── Images
│   ├── ache-adult-depression-expression-41253.jpg
│   ├── ...
├── test.txt
├── train.txt
└── val.txt
```

### Evaluation

Run evaluation with the following command :

```shell
python eval.py -m pphumanseg -d mini_supervisely -dr /path/to/pphumanseg
```

Run evaluation on quantized model with the following command :

```shell
python eval.py -m pphumanseg_q -d mini_supervisely -dr /path/to/pphumanseg
```