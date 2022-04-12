# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import collections
import numpy as numpy
import cv2 as cv

class Compose:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Resize:
    def __init__(self, size, interpolation=cv.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return cv.resize(img, self.size)

class CenterCrop:
    def __init__(self, size):
        self.size = size # w, h

    def __call__(self, img):
        h, w, _ = img.shape
        ws = int(w / 2 - self.size[0] / 2)
        hs = int(h / 2 - self.size[1] / 2)
        return img[hs:hs+self.size[1], ws:ws+self.size[0], :]

class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if self.mean is not None:
            img[:, :, 0] = img[:, :, 0] - self.mean[0]
            img[:, :, 1] = img[:, :, 1] - self.mean[1]
            img[:, :, 2] = img[:, :, 2] - self.mean[2]
        if self.std is not None:
            img[:, :, 0] = img[:, :, 0] / self.std[0]
            img[:, :, 1] = img[:, :, 1] / self.std[1]
            img[:, :, 2] = img[:, :, 2] / self.std[2]
        return img

class ColorConvert:
    def __init__(self, ctype):
        self.ctype = ctype

    def __call__(self, img):
        return cv.cvtColor(img, self.ctype)
