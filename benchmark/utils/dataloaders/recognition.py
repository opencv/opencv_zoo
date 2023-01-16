import os

import numpy as np
import cv2 as cv

from .base_dataloader import _BaseImageLoader
from ..factory import DATALOADERS

@DATALOADERS.register
class RecognitionImageLoader(_BaseImageLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._labels = self._load_label()

    def _load_label(self):
        labels = dict.fromkeys(self._files, None)
        for filename in self._files:
            if os.path.exists(os.path.join(self._path, '{}.txt'.format(filename[:-4]))):
                labels[filename] = np.loadtxt(os.path.join(self._path, '{}.txt'.format(filename[:-4])), ndmin=2)
            else:
                labels[filename] = None
        return labels

    def __iter__(self):
        for filename in self._files:
            image = cv.imread(os.path.join(self._path, filename))
            if [0, 0] in self._sizes:
                yield filename, image, self._labels[filename]
            else:
                for size in self._sizes:
                    image_r = cv.resize(image, size)
                    yield filename, image_r, self._labels[filename]