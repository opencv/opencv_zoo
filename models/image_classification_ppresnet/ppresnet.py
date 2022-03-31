# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.


import numpy as np
import cv2 as cv

class PPResNet:
    def __init__(self, modelPath, labelPath, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._labelPath = labelPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.dnn.readNet(self._modelPath)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        self._inputNames = ''
        self._outputNames = ['save_infer_model/scale_0.tmp_0']
        self._inputSize = [224, 224]
        self._mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
        self._std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]

        # load labels
        self._labels = self._load_labels()

    def _load_labels(self):
        labels = []
        with open(self._labelPath, 'r') as f:
            for line in f:
                labels.append(line.strip())
        return labels

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backend_id):
        self._backendId = backend_id
        self._model.setPreferableBackend(self._backendId)

    def setTarget(self, target_id):
        self._targetId = target_id
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image):
        image = image.astype(np.float32, copy=False) / 255.0
        image -= self._mean
        image /= self._std
        return cv.dnn.blobFromImage(image)

    def infer(self, image):
        assert image.shape[0] == self._inputSize[1], '{} (height of input image) != {} (preset height)'.format(image.shape[0], self._inputSize[1])
        assert image.shape[1] == self._inputSize[0], '{} (width of input image) != {} (preset width)'.format(image.shape[1], self._inputSize[0])

        # Preprocess
        inputBlob = self._preprocess(image)

        # Forward
        self._model.setInput(inputBlob, self._inputNames)
        outputBlob = self._model.forward(self._outputNames)

        # Postprocess
        results = self._postprocess(outputBlob)

        return results

    def _postprocess(self, outputBlob):
        class_id = np.argmax(outputBlob[0])
        return self._labels[class_id]

