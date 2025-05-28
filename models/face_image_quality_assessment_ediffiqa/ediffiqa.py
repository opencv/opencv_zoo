# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.

import numpy as np
import cv2 as cv


class eDifFIQA:

    def __init__(self, modelPath, inputSize=[112, 112]):
        self.modelPath = modelPath
        self.inputSize = tuple(inputSize) # [w, h]

        self.model = cv.dnn.readNetFromONNX(self.modelPath)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self.model.setPreferableBackend(self._backendId)
        self.model.setPreferableTarget(self._targetId)

    def infer(self, image):
        # Preprocess image
        image = self._preprocess(image)
        # Forward
        self.model.setInput(image)
        quality_score = self.model.forward()
        
        return quality_score

    def _preprocess(self, image: cv.Mat):
        # Change image from BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Resize to (112, 112)
        image = cv.resize(image, self.inputSize)
        # Scale to [0, 1] and normalize by mean=0.5, std=0.5
        image = ((image / 255) - 0.5) / 0.5
        # Move channel axis
        image = np.moveaxis(image[None, ...], -1, 1)

        return image
