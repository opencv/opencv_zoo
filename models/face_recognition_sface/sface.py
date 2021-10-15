# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv

from _testcapi import FLT_MIN

class SFace:
    def __init__(self, modelPath):
        self._model = cv.dnn.readNet(modelPath)
        self._input_size = [112, 112]
        self._dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        self._dst_mean = np.array([56.0262, 71.9008], dtype=np.float32)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backend_id):
        self._model.setPreferableBackend(backend_id)

    def setTarget(self, target_id):
        self._model.setPreferableTarget(target_id)

    def _preprocess(self, image, bbox):
        aligned_image = self._alignCrop(image, bbox)
        return cv.dnn.blobFromImage(aligned_image)

    def infer(self, image, bbox):
        # Preprocess
        inputBlob = self._preprocess(image, bbox)

        # Forward
        self._model.setInput(inputBlob)
        outputBlob = self._model.forward()

        # Postprocess
        results = self._postprocess(outputBlob)

        return results

    def _postprocess(self, outputBlob):
        return outputBlob / cv.norm(outputBlob)

    def match(self, image1, face1, image2, face2, dis_type=0):
        feature1 = self.infer(image1, face1)
        feature2 = self.infer(image2, face2)

        if dis_type == 0: # COSINE
            return np.sum(feature1 * feature2)
        elif dis_type == 1: # NORM_L2
            return cv.norm(feature1, feature2)
        else:
            raise NotImplementedError()

    def _alignCrop(self, image, face):
        # Retrieve landmarks
        if face.shape[-1] == (4 + 5 * 2):
            landmarks = face[4:].reshape(5, 2)
        else:
            raise NotImplementedError()
        warp_mat = self._getSimilarityTransformMatrix(landmarks)
        aligned_image = cv.warpAffine(image, warp_mat, self._input_size, flags=cv.INTER_LINEAR)
        return aligned_image

    def _getSimilarityTransformMatrix(self, src):
        # compute the mean of src and dst
        src_mean = np.array([np.mean(src[:, 0]), np.mean(src[:, 1])], dtype=np.float32)
        dst_mean = np.array([56.0262, 71.9008], dtype=np.float32)
        # subtract the means from src and dst
        src_demean = src.copy()
        src_demean[:, 0] = src_demean[:, 0] - src_mean[0]
        src_demean[:, 1] = src_demean[:, 1] - src_mean[1]
        dst_demean = self._dst.copy()
        dst_demean[:, 0] = dst_demean[:, 0] - dst_mean[0]
        dst_demean[:, 1] = dst_demean[:, 1] - dst_mean[1]

        A = np.array([[0., 0.], [0., 0.]], dtype=np.float64)
        for i in range(5):
            A[0][0] += dst_demean[i][0] * src_demean[i][0]
            A[0][1] += dst_demean[i][0] * src_demean[i][1]
            A[1][0] += dst_demean[i][1] * src_demean[i][0]
            A[1][1] += dst_demean[i][1] * src_demean[i][1]
        A = A / 5

        d = np.array([1.0, 1.0], dtype=np.float64)
        if A[0][0] * A[1][1] - A[0][1] * A[1][0] < 0:
            d[1] = -1

        T = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        s, u, vt = cv.SVDecomp(A)
        smax = s[0][0] if s[0][0] > s[1][0] else s[1][0]
        tol = smax * 2 * FLT_MIN
        rank = int(0)
        if s[0][0] > tol:
            rank += 1
        if s[1][0] > tol:
            rank += 1
        det_u = u[0][0] * u[1][1] - u[0][1] * u[1][0]
        det_vt = vt[0][0] * vt[1][1] - vt[0][1] * vt[1][0]
        if rank == 1:
            if det_u * det_vt > 0:
                uvt = np.matmul(u, vt)
                T[0][0] = uvt[0][0]
                T[0][1] = uvt[0][1]
                T[1][0] = uvt[1][0]
                T[1][1] = uvt[1][1]
            else:
                temp = d[1]
                d[1] = -1
                D = np.array([[d[0], 0.0], [0.0, d[1]]], dtype=np.float64)
                Dvt = np.matmul(D, vt)
                uDvt = np.matmul(u, Dvt)
                T[0][0] = uDvt[0][0]
                T[0][1] = uDvt[0][1]
                T[1][0] = uDvt[1][0]
                T[1][1] = uDvt[1][1]
                d[1] = temp
        else:
            D = np.array([[d[0], 0.0], [0.0, d[1]]], dtype=np.float64)
            Dvt = np.matmul(D, vt)
            uDvt = np.matmul(u, Dvt)
            T[0][0] = uDvt[0][0]
            T[0][1] = uDvt[0][1]
            T[1][0] = uDvt[1][0]
            T[1][1] = uDvt[1][1]

        var1 = 0.0
        var2 = 0.0
        for i in range(5):
            var1 += src_demean[i][0] * src_demean[i][0]
            var2 += src_demean[i][1] * src_demean[i][1]
        var1 /= 5
        var2 /= 5

        scale = 1.0 / (var1 + var2) * (s[0][0] * d[0] + s[1][0] * d[1])
        TS = [
            T[0][0] * src_mean[0] + T[0][1] * src_mean[1],
            T[1][0] * src_mean[0] + T[1][1] * src_mean[1]
        ]
        T[0][2] = dst_mean[0] - scale * TS[0]
        T[1][2] = dst_mean[1] - scale * TS[1]
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale
        return np.array([
            [T[0][0], T[0][1], T[0][2]],
            [T[1][0], T[1][1], T[1][2]]
        ], dtype=np.float64)