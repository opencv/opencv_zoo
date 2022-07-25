import cv2
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

class NanoDet():
    def __init__(self, prob_threshold=0.35, iou_threshold=0.6):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        self.num_classes = len(self.classes)
        self.strides = (8, 16, 32, 64)
        self.image_shape = (416, 416)
        self.reg_max = 7
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNet('object_detection_nanodet-plus-m-1.5x-416.onnx')

        self.anchors_mlvl = []
        for i in range(len(self.strides)):
            featmap_size = (int(self.image_shape[0] / self.strides[i]), int(self.image_shape[1] / self.strides[i]))
            stride = self.strides[i]
            feat_h, feat_w = featmap_size
            shift_x = np.arange(0, feat_w) * stride
            shift_y = np.arange(0, feat_h) * stride
            xv, yv = np.meshgrid(shift_x, shift_y)
            xv = xv.flatten()
            yv = yv.flatten()
            cx = xv + 0.5 * (stride-1)
            cy = yv + 0.5 * (stride - 1)
            anchors = np.stack((cx, cy), axis=-1)
            self.anchors_mlvl.append(anchors)

    def softmax_func(self,x, axis=0):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def pre_process(self, img):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        blob = cv2.dnn.blobFromImage(img)
        return blob

    def infer(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.image_shape[0], self.image_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.image_shape[0], int(self.image_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.image_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.image_shape[1] - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.image_shape[0] * hw_scale), self.image_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.image_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.image_shape[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, self.image_shape, interpolation=cv2.INTER_AREA)

        blob = self.pre_process(img)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        det_bboxes, det_conf, det_classid = self.post_process(outs)

        drawimg = srcimg.copy()
        ratioh,ratiow = srcimg.shape[0]/newh,srcimg.shape[1]/neww
        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = max(int((det_bboxes[i,0] - left) * ratiow), 0), max(int((det_bboxes[i,1] - top) * ratioh), 0), min(
                int((det_bboxes[i,2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i,3] - top) * ratioh), srcimg.shape[0])
            classId = det_classid[i]
            conf = det_conf[i]
            left = xmin
            top = ymin
            right = xmax
            bottom = ymax
            cv2.rectangle(drawimg, (left, top), (right, bottom), (0, 0, 0), thickness=2)
            #label = '%.2f' % conf
            label =''
            label = '%s%s' % (self.classes[classId], label)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
            cv2.putText(drawimg, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return drawimg

    def post_process(self, preds):
        cls_scores, bbox_preds = preds[::2], preds[1::2]
        rescale = False
        scale_factor = 1
        bboxes_mlvl = []
        scores_mlvl = []
        for stride, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.anchors_mlvl):
            if cls_score.ndim==3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim==3:
                bbox_pred = bbox_pred.squeeze(axis=0)
            bbox_pred = self.softmax_func(bbox_pred.reshape(-1, self.reg_max + 1), axis=1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1,4)
            bbox_pred *= stride

            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]

            points = anchors
            distance = bbox_pred
            max_shape=self.image_shape
            x1 = points[:, 0] - distance[:, 0]
            y1 = points[:, 1] - distance[:, 1]
            x2 = points[:, 0] + distance[:, 2]
            y2 = points[:, 1] + distance[:, 3]

            if max_shape is not None:
                x1 = np.clip(x1, 0, max_shape[1])
                y1 = np.clip(y1, 0, max_shape[0])
                x2 = np.clip(x2, 0, max_shape[1])
                y2 = np.clip(y2, 0, max_shape[0])

            bboxes = np.stack([x1, y1, x2, y2], axis=-1)
            bboxes_mlvl.append(bboxes)
            scores_mlvl.append(cls_score)

        bboxes_mlvl = np.concatenate(bboxes_mlvl, axis=0)
        if rescale:
            bboxes_mlvl /= scale_factor
        scores_mlvl = np.concatenate(scores_mlvl, axis=0)
        bboxes_wh = bboxes_mlvl.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]
        classIds = np.argmax(scores_mlvl, axis=1)
        confidences = np.max(scores_mlvl, axis=1)

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)

        if len(indices)>0:
            det_bboxes = bboxes_mlvl[indices[:]]
            det_conf = confidences[indices[:]]
            det_classid = classIds[indices[:]]

        else:
            det_bboxes = np.array([])
            det_conf = np.array([])
            det_classid = np.array([])

        return det_bboxes.astype(np.float32), det_conf, det_classid
