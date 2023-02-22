# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import collections
import numpy as np
import cv2 as cv
import sys

class Compose:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
            if img is None:
                break
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
        img = img.astype("float32")
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

class HandAlign:
    def __init__(self, model):
        self.model = model
        sys.path.append('../../models/palm_detection_mediapipe')
        from mp_palmdet import MPPalmDet
        self.palm_detector = MPPalmDet(modelPath='../../models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx', nmsThreshold=0.3, scoreThreshold=0.9)

    def __call__(self, img):
        return self.mp_handpose_align(img)

    def mp_handpose_align(self, img):
        palms = self.palm_detector.infer(img)
        if len(palms) == 0:
            return None
        palm = palms[0]
        palm_bbox = palm[0:4].reshape(2, 2)
        palm_landmarks = palm[4:18].reshape(7, 2)
        p1 = palm_landmarks[0]
        p2 = palm_landmarks[2]
        radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        radians = radians - 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)
        #  get bbox center
        center_palm_bbox = np.sum(palm_bbox, axis=0) / 2
        #  get rotation matrix
        rotation_matrix = cv.getRotationMatrix2D(center_palm_bbox, angle, 1.0)
        #  get rotated image
        rotated_image = cv.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        #  get bounding boxes from rotated palm landmarks
        homogeneous_coord = np.c_[palm_landmarks, np.ones(palm_landmarks.shape[0])]
        rotated_palm_landmarks = np.array([
            np.dot(homogeneous_coord, rotation_matrix[0]),
            np.dot(homogeneous_coord, rotation_matrix[1])])
        #  get landmark bounding box
        rotated_palm_bbox = np.array([
            np.amin(rotated_palm_landmarks, axis=1),
            np.amax(rotated_palm_landmarks, axis=1)])  # [top-left, bottom-right]

        #  shift bounding box
        wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        shift_vector = [0, -0.1] * wh_rotated_palm_bbox
        rotated_palm_bbox = rotated_palm_bbox + shift_vector
        #  squarify bounding boxx
        center_rotated_plam_bbox = np.sum(rotated_palm_bbox, axis=0) / 2
        wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        new_half_size = np.amax(wh_rotated_palm_bbox) / 2
        rotated_palm_bbox = np.array([
            center_rotated_plam_bbox - new_half_size,
            center_rotated_plam_bbox + new_half_size])

        #  enlarge bounding box
        center_rotated_plam_bbox = np.sum(rotated_palm_bbox, axis=0) / 2
        wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        new_half_size = wh_rotated_palm_bbox * 1.5
        rotated_palm_bbox = np.array([
            center_rotated_plam_bbox - new_half_size,
            center_rotated_plam_bbox + new_half_size])

        # Crop the rotated image by the bounding box
        [[x1, y1], [x2, y2]] = rotated_palm_bbox.astype(np.int32)
        diff = np.maximum([-x1, -y1, x2 - rotated_image.shape[1], y2 - rotated_image.shape[0]], 0)
        [x1, y1, x2, y2] = [x1, y1, x2, y2] + diff
        crop = rotated_image[y1:y2, x1:x2, :]
        crop = cv.copyMakeBorder(crop, diff[1], diff[3], diff[0], diff[2], cv.BORDER_CONSTANT, value=(0, 0, 0))
        return crop
