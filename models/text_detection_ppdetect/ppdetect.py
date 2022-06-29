# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.
import cv2
import numpy as np
import cv2 as cv
from shapely.geometry import Polygon
import pyclipper
class ppdetect:
    def __init__(self, modelPath, boxThresh=0.6, binaryThreshold=0.3, isPoly=True, maxCandidates=200,minSize=3 ,unclipRatio=2.0, backendId=0, targetId=0):
        self.model_path = modelPath
        self.backend_id = backendId
        self.target_id = targetId
        self.input_names = ''
        self.output_names = ''

        self.model = cv.dnn.readNet(self.model_path)
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.binaryThreshold = binaryThreshold
        self.boxThresh = boxThresh
        self.maxCandidates =maxCandidates
        self.isPoly = isPoly
        self.unclipRatio = unclipRatio
        self.minSize = minSize
    @property
    def name(self):
        return self.__class__.__name__

    def _preprocess(self, image):
        input_blob = (image / 255.0 - self.mean) / self.std
        input_blob = input_blob.transpose(2, 0, 1)
        input_blob = input_blob[np.newaxis, :, :, :]
        input_blob = input_blob.astype(np.float32)
        return input_blob

    def infer(self, image):
        # Preprocess
        input_blob = self._preprocess(image)
        self.model.setInput(input_blob, self.input_names)
        # Forward
        output_blob = self.model.forward(self.output_names)
        # Postprocess
        results = self._postprocess(output_blob)

        return results

    def polygonsFromBitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        pred = pred
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.maxCandidates]:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.boxScoreFast(pred, points.reshape(-1, 2))
            if self.boxThresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unClip(points, self.unclipRatio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.getMiniBoxes(box.reshape((-1, 1, 2)))
            if sside < self.minSize + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxesFromBitmap(self, pred, _bitmap, destWidth, destHeight):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.maxCandidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int32)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.getMiniBoxes(contour)
            if sside < self.minSize:
                continue
            points = np.array(points)
            score = self.boxScoreFast(pred, points.reshape(-1, 2))
            if self.boxThresh > score:
                continue

            box = self.unClip(points, self.unclipRatio).reshape(-1, 1, 2)
            box, sside = self.getMiniBoxes(box)
            if sside < self.minSize + 2:
                continue
            box = np.array(box)
            if not isinstance(destWidth, int):
                destWidth = destWidth.item()
                destHeight = destHeight.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * destWidth), 0, destWidth)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * destHeight), 0, destHeight)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unClip(self, box, unclip_ratio=2):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def getMiniBoxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index1, index2, index3, index = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index1 = 0
            index4 = 1
        else:
            index1 = 1
            index4 = 0
        if points[3][1] > points[2][1]:
            index2 = 2
            index3 = 3
        else:
            index2 = 3
            index3 = 2

        box = [
            points[index1], points[index2], points[index3], points[index4]
        ]
        return box, min(bounding_box[1])

    def boxScoreFast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def _postprocess(self, pred):
        pred = pred[:, 0, :, :]
        segmentation = pred > self.binaryThreshold

        boxes_batch = []
        score_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[-2:]
            if (self.isPoly):
                tmp_boxes, tmp_scores = self.polygonsFromBitmap(
                    pred[batch_index], segmentation[batch_index], width, height)

                boxes = []
                score = []
                for k in range(len(tmp_boxes)):
                    if tmp_scores[k] > self.boxThresh:
                        boxes.append(tmp_boxes[k])
                        score.append(tmp_scores[k])
                if len(boxes) > 0:
                    for i in range(len(boxes)):
                        boxes[i] = np.array(boxes[i])
                        boxes[i][:, 0] = boxes[i][:, 0] * 1.0
                        boxes[i][:, 1] = boxes[i][:, 1] * 1.0

                boxes_batch.append(boxes)
                score_batch.append(score)
            else:
                tmp_boxes, tmp_scores = self.boxesFromBitmap(
                    pred[batch_index], segmentation[batch_index], width, height)

                boxes = []
                score = []
                for k in range(len(tmp_boxes)):
                    if tmp_scores[k] > self.boxThresh:
                        boxes.append(tmp_boxes[k])
                        score.append(tmp_scores[k])
                if len(boxes) > 0:
                    boxes = np.array(boxes)
                    boxes[:, :, 0] = boxes[:, :, 0] * 1.0
                    boxes[:, :, 1] = boxes[:, :, 1] * 1.0
                boxes_batch.append(boxes)
                score_batch.append(score)

        return tuple(boxes_batch[0]), np.array(score_batch[0])

