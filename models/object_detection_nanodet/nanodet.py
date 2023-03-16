import numpy as np
import cv2

class NanoDet:
    def __init__(self, modelPath, prob_threshold=0.35, iou_threshold=0.6, backend_id=0, target_id=0):
        self.strides = (8, 16, 32, 64)
        self.image_shape = (416, 416)
        self.reg_max = 7
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.backend_id = backend_id
        self.target_id = target_id
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNet(modelPath)
        self.net.setPreferableBackend(self.backend_id)
        self.net.setPreferableTarget(self.target_id)

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
            #anchors = np.stack((cx, cy), axis=-1)
            anchors = np.column_stack((cx, cy))
            self.anchors_mlvl.append(anchors)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self.net.setPreferableBackend(self.backend_id)
        self.net.setPreferableTarget(self.target_id)

    def pre_process(self, img):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        blob = cv2.dnn.blobFromImage(img)
        return blob

    def infer(self, srcimg):
        blob = self.pre_process(srcimg)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        preds = self.post_process(outs)
        return preds

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

            x_exp = np.exp(bbox_pred.reshape(-1, self.reg_max + 1))
            x_sum = np.sum(x_exp, axis=1, keepdims=True)
            bbox_pred = x_exp / x_sum
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

            #bboxes = np.stack([x1, y1, x2, y2], axis=-1)
            bboxes = np.column_stack([x1, y1, x2, y2])
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
            det_bboxes = bboxes_mlvl[indices]
            det_conf = confidences[indices]
            det_classid = classIds[indices]

            return np.concatenate([det_bboxes, det_conf.reshape(-1, 1), det_classid.reshape(-1, 1)], axis=1)
        else:
            return np.array([])
