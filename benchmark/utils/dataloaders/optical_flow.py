import os

import numpy as np
import cv2 as cv

from .base_dataloader import _BaseImageLoader
from ..factory import DATALOADERS

@DATALOADERS.register
class OpticalFlowImageLoader(_BaseImageLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        for case in self._files:
            image0 = cv.imread(os.path.join(self._path, case[0]))
            image1 = cv.imread(os.path.join(self._path, case[1]))
            if [0, 0] in self._sizes:
                yield "{}, {}".format(case[0], case[1]), image0, image1
            else:
                for size in self._sizes:
                    image0_r = cv.resize(image0, size)
                    image1_r = cv.resize(image1, size)
                    yield "{}, {}".format(case[0], case[1]), image0_r, image1_r