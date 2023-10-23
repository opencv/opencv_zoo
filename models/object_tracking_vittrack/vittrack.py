# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.

import numpy as np
import cv2 as cv

class VitTrack:
    def __init__(self, model_path, backend_id=0, target_id=0):
        self.model_path = model_path
        self.backend_id = backend_id
        self.target_id = target_id

        self.params = cv.TrackerVit_Params()
        self.params.net = self.model_path
        self.params.backend = self.backend_id
        self.params.target = self.target_id

        self.model = cv.TrackerVit_create(self.params)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backend_id, target_id):
        self.backend_id = backend_id
        self.target_id = target_id

        self.params.backend = self.backend_id
        self.params.target = self.target_id

        self.model = cv.TrackerVit_create(self.params)

    def init(self, image, roi):
        self.model.init(image, roi)

    def infer(self, image):
        is_located, bbox = self.model.update(image)
        score = self.model.getTrackingScore()
        return is_located, bbox, score
