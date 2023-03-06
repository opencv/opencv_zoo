import os
import numpy as np
import cv2 as cv
import xml.dom.minidom as minidom
from tqdm import tqdm

class ICDAR:
    def __init__(self, root):
        self.root = root
        self.acc = -1
        self.inputSize = [100, 32]
        self.val_label_file = os.path.join(root, "word.xml")
        self.val_label = self.load_label(self.val_label_file)

    @property
    def name(self):
        return self.__class__.__name__

    def load_label(self, label_file):
        label = list()
        dom = minidom.getDOMImplementation().createDocument(None, 'Root', None)
        root = dom.documentElement
        dom = minidom.parse(self.val_label_file)
        root = dom.documentElement
        names = root.getElementsByTagName('image')
        for name in names:
            key = os.path.join(self.root, name.getAttribute('file'))
            value = name.getAttribute('tag').lower()
            label.append([key, value])

        return label

    def eval(self, model):
        right_num = 0
        pbar = tqdm(self.val_label)
        pbar.set_description("Evaluating {} with {} val set".format(model.name, self.name))

        for fn, label in pbar:

            img = cv.imread(fn)

            rbbox = np.array([0, img.shape[0], 0, 0, img.shape[1], 0, img.shape[1], img.shape[0]])
            pred = model.infer(img, rbbox)
            if label.lower() == pred.lower():
                right_num += 1

        self.acc = right_num/(len(self.val_label) * 1.0)


    def get_result(self):
        return self.acc

    def print_result(self):
        print("Accuracy: {:.2f}%".format(self.acc*100))