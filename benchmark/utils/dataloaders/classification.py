import os

import numpy as np
import cv2 as cv

from .base_dataloader import _BaseImageLoader
from ..factory import DATALOADERS

@DATALOADERS.register
class ClassificationImageLoader(_BaseImageLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._to_rgb = kwargs.pop('toRGB', False)
        self._center_crop = kwargs.pop('centerCrop', None)

    def _toRGB(self, image):
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)

    def _centerCrop(self, image):
        h, w, _ = image.shape
        w_crop = int((w - self._center_crop) / 2.)
        assert w_crop >= 0
        h_crop = int((h - self._center_crop) / 2.)
        assert h_crop >= 0
        return image[w_crop:w-w_crop, h_crop:h-h_crop, :]

    def __iter__(self):
        for filename in self._files:
            image = cv.imread(os.path.join(self._path, filename))

            if self._to_rgb:
                image = self._toRGB(image)

            if [0, 0] in self._sizes:
                yield filename, image
            else:
                for size in self._sizes:
                    image = cv.resize(image, size)
                    if self._center_crop:
                        image = self._centerCrop(image)
                    yield filename, image