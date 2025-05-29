import numpy as np
import cv2 as cv

class EfficientSAM:
    def __init__(self, modelPath, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.dnn.readNet(self._modelPath)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)
        # 3 inputs
        self._inputNames = ["batched_images", "batched_point_coords", "batched_point_labels"]

        self._outputNames = ['output_masks', 'iou_predictions']  # actual output layer name
        self._currentInputSize = None
        self._inputSize = [1024, 1024]  # input size for the model
        self._maxPointNums = 6
        self._frontGroundPoints = []
        self._backGroundPoints = []
        self._labels = []

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image, points, labels):

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # record the input image size, (width, height)
        self._currentInputSize = (image.shape[1], image.shape[0])

        image = cv.resize(image, self._inputSize)

        image = image.astype(np.float32, copy=False) / 255.0

        image_blob = cv.dnn.blobFromImage(image)

        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        assert points.shape[0] <= self._maxPointNums, f"Max input points number: {self._maxPointNums}"
        assert points.shape[0] == labels.shape[0]

        frontGroundPoints = []
        backGroundPoints = []
        inputLabels = []
        for i in range(len(points)):
            if labels[i] == -1:
                backGroundPoints.append(points[i])
            else:
                frontGroundPoints.append(points[i])
                inputLabels.append(labels[i])
        self._backGroundPoints = np.uint32(backGroundPoints)
        # print("input:")
        # print(" back: ", self._backGroundPoints)
        # print(" front: ", frontGroundPoints)
        # print(" label: ", inputLabels)

        # convert points to (1024*1024) size space
        for p in frontGroundPoints:
            p[0] = np.float32(p[0] * self._inputSize[0]/self._currentInputSize[0])
            p[1] = np.float32(p[1] * self._inputSize[1]/self._currentInputSize[1])

        if len(frontGroundPoints) > self._maxPointNums:
            return "no"

        pad_num = self._maxPointNums - len(frontGroundPoints)
        self._frontGroundPoints = np.vstack([frontGroundPoints, np.zeros((pad_num, 2), dtype=np.float32)])
        inputLabels_arr = np.array(inputLabels, dtype=np.float32).reshape(-1, 1)
        self._labels = np.vstack([inputLabels_arr, np.full((pad_num, 1), -1, dtype=np.float32)])

        points_blob = np.array([[self._frontGroundPoints]])

        labels_blob = np.array([[self._labels]])

        return image_blob, points_blob, labels_blob

    def infer(self, image, points, labels):
        # Preprocess
        imageBlob, pointsBlob, labelsBlob = self._preprocess(image, points, labels)
        # Forward
        self._model.setInput(imageBlob, self._inputNames[0])
        self._model.setInput(pointsBlob, self._inputNames[1])
        self._model.setInput(labelsBlob, self._inputNames[2])
        # print("infering...")
        outputs = self._model.forward(self._outputNames)
        outputBlob, outputIou = outputs[0], outputs[1]
        # Postprocess
        results = self._postprocess(outputBlob, outputIou)
        # print("done")
        return results

    def _postprocess(self, outputBlob, outputIou):
        # The masks are already sorted by their predicted IOUs.
        # The first dimension is the batch size (we have a single image. so it is 1).
        # The second dimension is the number of masks we want to generate
        # The third dimension is the number of candidate masks output by the model.
        masks = outputBlob[0, 0, :, :, :] >= 0
        ious = outputIou[0, 0, :]

        # sorted by ious
        sorted_indices = np.argsort(ious)[::-1]
        sorted_masks = masks[sorted_indices]

        # sorted by area
        # mask_areas = np.sum(masks, axis=(1, 2))
        # sorted_indices = np.argsort(mask_areas)
        # sorted_masks = masks[sorted_indices]

        masks_uint8 = (sorted_masks * 255).astype(np.uint8)

        # change to real image size
        resized_masks = [
            cv.resize(mask, dsize=self._currentInputSize,
                    interpolation=cv.INTER_NEAREST)
            for mask in masks_uint8
        ]

        # background mask don't need
        for mask in resized_masks:
            contains_bg = any(
                mask[y, x] if (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0])
                else False
                for (x, y) in self._backGroundPoints
            )
            if not contains_bg:
                return mask

        return resized_masks[0]
