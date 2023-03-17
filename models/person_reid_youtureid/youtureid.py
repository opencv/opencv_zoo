# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv

class YoutuReID:
    def __init__(self, modelPath, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.dnn.readNet(modelPath)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        self._input_size = (128, 256) # fixed
        self._output_dim = 768
        self._mean = (0.485, 0.456, 0.406)
        self._std = (0.229, 0.224, 0.225)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image):
        image = image[:, :, ::-1]
        image = (image / 255.0 - self._mean) / self._std
        return cv.dnn.blobFromImage(image.astype(np.float32))
        # return cv.dnn.blobFromImage(image, scalefactor=(1.0/255.0), size=self._input_size, mean=self._mean) / self._std

    def infer(self, image):
        # Preprocess
        inputBlob = self._preprocess(image)

        # Forward
        self._model.setInput(inputBlob)
        features = self._model.forward()
        return np.reshape(features, (features.shape[0], features.shape[1]))

    def query(self, query_img_list, gallery_img_list, topK=5):
        query_features_list = []
        for q in query_img_list:
            query_features_list.append(self.infer(q))
        query_features = np.concatenate(query_features_list, axis=0)
        query_norm = np.linalg.norm(query_features, ord=2, axis=1, keepdims=True)
        query_arr = query_features / (query_norm + np.finfo(np.float32).eps)

        gallery_features_list = []
        for g in gallery_img_list:
            gallery_features_list.append(self.infer(g))
        gallery_features = np.concatenate(gallery_features_list, axis=0)
        gallery_norm = np.linalg.norm(gallery_features, ord=2, axis=1, keepdims=True)
        gallery_arr = gallery_features / (gallery_norm + np.finfo(np.float32).eps)

        dist = np.matmul(query_arr, gallery_arr.T)
        idx = np.argsort(-dist, axis=1)
        return [i[0:topK] for i in idx]
