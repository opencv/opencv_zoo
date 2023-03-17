# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2022, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv

class FacialExpressionRecog:
    def __init__(self, modelPath, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.dnn.readNet(self._modelPath)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

        self._align_model = FaceAlignment()

        self._inputNames = 'data'
        self._outputNames = ['label']
        self._inputSize = [112, 112]
        self._mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        self._std = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image, bbox):
        if bbox is not None:
            image = self._align_model.get_align_image(image, bbox[4:].reshape(-1, 2))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image.astype(np.float32, copy=False) / 255.0
        image -= self._mean
        image /= self._std
        return cv.dnn.blobFromImage(image)

    def infer(self, image, bbox=None):
        # Preprocess
        inputBlob = self._preprocess(image, bbox)

        # Forward
        self._model.setInput(inputBlob, self._inputNames)
        outputBlob = self._model.forward(self._outputNames)

        # Postprocess
        results = self._postprocess(outputBlob)

        return results

    def _postprocess(self, outputBlob):
        result = np.argmax(outputBlob[0], axis=1).astype(np.uint8)
        return result

    @staticmethod
    def getDesc(ind):
        _expression_enum = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
        return _expression_enum[ind]


class FaceAlignment():
    def __init__(self, reflective=False):
        self._std_points = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
        self.reflective = reflective

    def __tformfwd(self, trans, uv):
        uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
        xy = np.dot(uv, trans)
        xy = xy[:, 0:-1]
        return xy

    def __tforminv(self, trans, uv):
        Tinv = np.linalg.inv(trans)
        xy = self.__tformfwd(Tinv, uv)
        return xy

    def __findNonreflectiveSimilarity(self, uv, xy, options=None):
        options = {"K": 2}

        K = options["K"]
        M = xy.shape[0]
        x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
        y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
        # print '--->x, y:\n', x, y

        tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
        tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
        X = np.vstack((tmp1, tmp2))
        # print '--->X.shape: ', X.shape
        # print 'X:\n', X

        u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
        v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
        U = np.vstack((u, v))
        # print '--->U.shape: ', U.shape
        # print 'U:\n', U

        # We know that X * r = U
        if np.linalg.matrix_rank(X) >= 2 * K:
            r, _, _, _ = np.linalg.lstsq(X, U, rcond=-1)
            # print(r, X, U, sep="\n")
            r = np.squeeze(r)
        else:
            raise Exception("cp2tform:twoUniquePointsReq")

        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]

        Tinv = np.array([[sc, -ss, 0], [ss, sc, 0], [tx, ty, 1]])
        T = np.linalg.inv(Tinv)
        T[:, 2] = np.array([0, 0, 1])

        return T, Tinv

    def __findSimilarity(self, uv, xy, options=None):
        options = {"K": 2}

        #    uv = np.array(uv)
        #    xy = np.array(xy)

        # Solve for trans1
        trans1, trans1_inv = self.__findNonreflectiveSimilarity(uv, xy, options)

        # manually reflect the xy data across the Y-axis
        xyR = xy
        xyR[:, 0] = -1 * xyR[:, 0]
        # Solve for trans2
        trans2r, trans2r_inv = self.__findNonreflectiveSimilarity(uv, xyR, options)

        # manually reflect the tform to undo the reflection done on xyR
        TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        trans2 = np.dot(trans2r, TreflectY)

        # Figure out if trans1 or trans2 is better
        xy1 = self.__tformfwd(trans1, uv)
        norm1 = np.linalg.norm(xy1 - xy)
        xy2 = self.__tformfwd(trans2, uv)
        norm2 = np.linalg.norm(xy2 - xy)

        if norm1 <= norm2:
            return trans1, trans1_inv
        else:
            trans2_inv = np.linalg.inv(trans2)
            return trans2, trans2_inv

    def __get_similarity_transform(self, src_pts, dst_pts):
        if self.reflective:
            trans, trans_inv = self.__findSimilarity(src_pts, dst_pts)
        else:
            trans, trans_inv = self.__findNonreflectiveSimilarity(src_pts, dst_pts)
        return trans, trans_inv

    def __cvt_tform_mat_for_cv2(self, trans):
        cv2_trans = trans[:, 0:2].T
        return cv2_trans

    def get_similarity_transform_for_cv2(self, src_pts, dst_pts):
        trans, trans_inv = self.__get_similarity_transform(src_pts, dst_pts)
        cv2_trans = self.__cvt_tform_mat_for_cv2(trans)
        return cv2_trans, trans

    def get_align_image(self, image, lm5_points):
        assert lm5_points is not None
        tfm, trans = self.get_similarity_transform_for_cv2(lm5_points, self._std_points)
        return cv.warpAffine(image, tfm, (112, 112))
