import os

import numpy as np
import cv2 as cv

from tqdm import tqdm

class ImageNet:
    def __init__(self, root, size=224):
        self.root = root
        self.size = size
        self.top1_acc = -1
        self.top5_acc = -1

        self.root_val = os.path.join(self.root, "ILSVRC", "Data", "CLS-LOC", "val")
        self.val_label_file = os.path.join(self.root, "caffe_ilsvrc12", "val.txt")

        self.val_label = self.load_label(self.val_label_file)

    @property
    def name(self):
        return self.__class__.__name__

    def load_label(self, label_file):
        label = list()
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                key, value = line.split()

                key = os.path.join(self.root_val, key)
                value = int(value)

                label.append([key, value])

        return label

    def eval(self, model):
        top_1_hits = 0
        top_5_hits = 0
        pbar = tqdm(self.val_label)
        pbar.set_description("Evaluating {} with {} val set".format(model.name, self.name))

        for fn, label in pbar:

            img = cv.imread(fn)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, dsize=(256, 256))
            img = img[16:240, 16:240, :]

            pred = model.infer(img)
            if label == pred[0][0]:
                top_1_hits += 1
            if label in pred[0]:
                top_5_hits += 1

        self.top1_acc = top_1_hits/(len(self.val_label) * 1.0)
        self.top5_acc = top_5_hits/(len(self.val_label) * 1.0)

    def get_result(self):
        return self.top1_acc, self.top5_acc

    def print_result(self):
        print("Top-1 Accuracy: {:.2f}%; Top-5 Accuracy: {:.2f}%".format(self.top1_acc*100, self.top5_acc*100))

