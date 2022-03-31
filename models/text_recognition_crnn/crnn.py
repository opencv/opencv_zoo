# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv

class CRNN:
    def __init__(self, modelPath, charsetPath, backendId=0, targetId=0):
        self._model_path = modelPath
        self._charsetPath = charsetPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.dnn.readNet(self._model_path)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        self._charset = self._load_charset(self._charsetPath)
        self._inputSize = [100, 32] # Fixed
        self._targetVertices = np.array([
            [0, self._inputSize[1] - 1],
            [0, 0],
            [self._inputSize[0] - 1, 0],
            [self._inputSize[0] - 1, self._inputSize[1] - 1]
        ], dtype=np.float32)

    @property
    def name(self):
        return self.__class__.__name__

    def _load_charset(self, charsetPath):
        charset = ''
        with open(charsetPath, 'r') as f:
            for char in f:
                char = char.strip()
                charset += char
        return charset

    def setBackend(self, backend_id):
        self._backendId = backend_id
        self._model.setPreferableBackend(self._backendId)

    def setTarget(self, target_id):
        self._targetId = target_id
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image, rbbox):
        # Remove conf, reshape and ensure all is np.float32
        vertices = rbbox.reshape((4, 2)).astype(np.float32)

        rotationMatrix = cv.getPerspectiveTransform(vertices, self._targetVertices)
        cropped = cv.warpPerspective(image, rotationMatrix, self._inputSize)

        if 'CN' in self._model_path:
            pass
        else:
            cropped = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

        return cv.dnn.blobFromImage(cropped, size=self._inputSize, mean=127.5, scalefactor=1 / 127.5)

    def infer(self, image, rbbox):
        # Preprocess
        inputBlob = self._preprocess(image, rbbox)

        # Forward
        self._model.setInput(inputBlob)
        outputBlob = self._model.forward()

        # Postprocess
        results = self._postprocess(outputBlob)

        return results

    def _postprocess(self, outputBlob):
        '''Decode charaters from outputBlob
        '''
        text = ''
        for i in range(outputBlob.shape[0]):
            c = np.argmax(outputBlob[i][0])
            if c != 0:
                text += self._charset[c - 1]
            else:
                text += '-'

        # adjacent same letters as well as background text must be removed to get the final output
        char_list = []
        for i in range(len(text)):
            if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
                char_list.append(text[i])
        return ''.join(char_list)

