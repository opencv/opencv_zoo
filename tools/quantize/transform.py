# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

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

class ColorConvert:
    def __init__(self, ctype):
        self.ctype = ctype

    def __call__(self, img):
        return cv.cvtColor(img, self.ctype)