# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv

class DaSiamRPN:
    def __init__(self, kernel_cls1_path, kernel_r1_path, model_path, backend_id=0, target_id=0):
        self._model_path = model_path
        self._kernel_cls1_path = kernel_cls1_path
        self._kernel_r1_path = kernel_r1_path
        self._backend_id = backend_id
        self._target_id = target_id

        self._param = cv.TrackerDaSiamRPN_Params()
        self._param.model = self._model_path
        self._param.kernel_cls1 = self._kernel_cls1_path
        self._param.kernel_r1 = self._kernel_r1_path
        self._param.backend = self._backend_id
        self._param.target = self._target_id
        self._model = cv.TrackerDaSiamRPN.create(self._param)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backend_id = backendId
        self._target_id = targetId

        self._param = cv.TrackerDaSiamRPN_Params()
        self._param.model = self._model_path
        self._param.kernel_cls1 = self._kernel_cls1_path
        self._param.kernel_r1 = self._kernel_r1_path
        self._param.backend = self._backend_id
        self._param.target = self._target_id
        self._model = cv.TrackerDaSiamRPN.create(self._param)

    def init(self, image, roi):
        self._model.init(image, roi)

    def infer(self, image):
        isLocated, bbox = self._model.update(image)
        score = self._model.getTrackingScore()
        return isLocated, bbox, score
