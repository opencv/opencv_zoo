from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
from sklearn.decomposition import PCA

import cv2 as cv
from tqdm import tqdm


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,
                 threshold_idx], fprs[fold_idx,
                                      threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


class LFW:
    def __init__(self, root, target_size=250):
        self.LFW_IMAGE_SIZE = 250

        self.lfw_root = root
        self.target_size = target_size

        self.lfw_pairs_path = os.path.join(self.lfw_root, 'view2/pairs.txt')
        self.image_path_pattern = os.path.join(self.lfw_root, 'lfw', '{person_name}', '{image_name}')

        self.lfw_image_paths, self.id_list = self.load_pairs()

    @property
    def name(self):
        return 'LFW'

    def __len__(self):
        return len(self.lfw_image_paths)

    @property
    def ids(self):
        return self.id_list

    def load_pairs(self):
        image_paths = []
        id_list = []
        with open(self.lfw_pairs_path, 'r') as f:
            for line in f.readlines()[1:]:
                line = line.strip().split()
                if len(line) == 3:
                    person_name = line[0]
                    image1_name = '{}_{:04d}.jpg'.format(person_name, int(line[1]))
                    image2_name = '{}_{:04d}.jpg'.format(person_name, int(line[2]))
                    image_paths += [
                        self.image_path_pattern.format(person_name=person_name, image_name=image1_name),
                        self.image_path_pattern.format(person_name=person_name, image_name=image2_name)
                    ]
                    id_list.append(True)
                elif len(line) == 4:
                    person1_name = line[0]
                    image1_name = '{}_{:04d}.jpg'.format(person1_name, int(line[1]))
                    person2_name = line[2]
                    image2_name = '{}_{:04d}.jpg'.format(person2_name, int(line[3]))
                    image_paths += [
                        self.image_path_pattern.format(person_name=person1_name, image_name=image1_name),
                        self.image_path_pattern.format(person_name=person2_name, image_name=image2_name)
                    ]
                    id_list.append(False)
        return image_paths, id_list

    def __getitem__(self, key):
        img = cv.imread(self.lfw_image_paths[key])
        if self.target_size != self.LFW_IMAGE_SIZE:
            img = cv.resize(img, (self.target_size, self.target_size))
        return img

    def eval(self, model):
        ids = self.ids
        embeddings = np.zeros(shape=(len(self), 128))
        face_bboxes = np.load("./datasets/lfw_face_bboxes.npy")
        for idx, img in tqdm(enumerate(self), desc="Evaluating {} with {} val set".format(model.name, self.name)):
            embedding = model.infer(img, face_bboxes[idx])
            embeddings[idx] = embedding

        embeddings = sklearn.preprocessing.normalize(embeddings)
        self.tpr, self.fpr, self.acc, self.val, self.std, self.far = evaluate(embeddings, ids, nrof_folds=10)
        self.acc, self.std = np.mean(self.acc), np.std(self.acc)

    def print_result(self):
        print("==================== Results ====================")
        print("Average Accuracy: {:.4f}".format(self.acc))
        print("=================================================")
