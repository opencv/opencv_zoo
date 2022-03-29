# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv

class DB:
    def __init__(self, modelPath, inputSize=[736, 736], binaryThreshold=0.3, polygonThreshold=0.5, maxCandidates=200, unclipRatio=2.0, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._model = cv.dnn_TextDetectionModel_DB(
            cv.dnn.readNet(self._modelPath)
        )

        self._inputSize = tuple(inputSize) # (w, h)
        self._inputHeight = inputSize[0]
        self._inputWidth = inputSize[1]
        self._binaryThreshold = binaryThreshold
        self._polygonThreshold = polygonThreshold
        self._maxCandidates = maxCandidates
        self._unclipRatio = unclipRatio
        self._backendId = backendId
        self._targetId = targetId

        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        self._model.setBinaryThreshold(self._binaryThreshold)
        self._model.setPolygonThreshold(self._polygonThreshold)
        self._model.setUnclipRatio(self._unclipRatio)
        self._model.setMaxCandidates(self._maxCandidates)

        self._model.setInputParams(1.0/255.0, self._inputSize, (122.67891434, 116.66876762, 104.00698793))

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backend):
        self._backendId = backend
        self._model.setPreferableBackend(self._backendId)

    def setTarget(self, target):
        self._targetId = target
        self._model.setPreferableTarget(self._targetId)

    def setInputSize(self, input_size):
        self._inputSize = tuple(input_size)
        self._model.setInputParams(1.0/255.0, self._inputSize, (122.67891434, 116.66876762, 104.00698793))

    def infer(self, image):
        assert image.shape[0] == self._inputSize[1], '{} (height of input image) != {} (preset height)'.format(image.shape[0], self._inputSize[1])
        assert image.shape[1] == self._inputSize[0], '{} (width of input image) != {} (preset width)'.format(image.shape[1], self._inputSize[0])

        return self._model.detect(image)

