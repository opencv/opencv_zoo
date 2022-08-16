import os

import numpy as np
import cv2 as cv

from .base_dataloader import _BaseImageLoader
from ..factory import DATALOADERS

@DATALOADERS.register
class EstimationImageLoader(_BaseImageLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._bboxes, self._landmarks = self._load_data()

    def _load_data(self):
        bboxes = dict.fromkeys(self._files, None)
        landmarks = dict.fromkeys(self._files, None)
        for filename in self._files:
            data = np.load(os.path.join(self._path, '{}.npz'.format(filename[:-4])))
            bboxes[filename] = data['bbox']
            landmarks[filename] = data['landmark']
        return bboxes, landmarks

    def __iter__(self):
        for filename in self._files:
            image = cv.imread(os.path.join(self._path, filename))
            if [0, 0] in self._sizes:
                yield filename, image, self._bboxes[filename], self._landmarks[filename]
            else:
                for size in self._sizes:
                    image_r = cv.resize(image, size)
                    yield filename, image_r, self._bboxes[filename], self._landmarks[filename]