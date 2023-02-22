import lmdb
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm

class IIIT5K:
    def __init__(self, root):
        self.root = root
        self.acc = -1
        self.inputSize = [100, 32]

        self.val_label = self.load_label(self.root)

    @property
    def name(self):
        return self.__class__.__name__

    def load_label(self, root):
        lmdb_file = root
        lmdb_env = lmdb.open(lmdb_file)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        label = list()
        for key, value in lmdb_cursor:
            image_index = key.decode()
            if image_index.split('-')[0] == 'image':
                img = cv.imdecode(np.fromstring(value, np.uint8), 3)
                label_index = 'label-' + image_index.split('-')[1]
                value = lmdb_txn.get(label_index.encode()).decode().lower()
                label.append([img, value])
            else:
                break
        return label

    def eval(self, model):
        right_num = 0
        pbar = tqdm(self.val_label)
        pbar.set_description("Evaluating {} with {} val set".format(model.name, self.name))

        for img, value in pbar:


            rbbox = np.array([0, img.shape[0], 0, 0, img.shape[1], 0, img.shape[1], img.shape[0]])
            pred = model.infer(img, rbbox).lower()
            if value == pred:
                right_num += 1

        self.acc = right_num/(len(self.val_label) * 1.0)


    def get_result(self):
        return self.acc

    def print_result(self):
        print("Accuracy: {:.2f}%".format(self.acc*100))