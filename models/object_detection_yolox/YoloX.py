import numpy as np
import cv2

class YoloX:
    def __init__(self, modelPath, confThreshold=0.35, nmsThreshold=0.5, objThreshold=0.5, backendId=0, targetId=0):
        self.num_classes = 80
        self.net = cv2.dnn.readNet(modelPath)
        self.input_size = (640, 640)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.strides = [8, 16, 32]
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.backendId = backendId
        self.targetId = targetId
        self.net.setPreferableBackend(self.backendId)
        self.net.setPreferableTarget(self.targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backenId):
        self.backendId = backendId
        self.net.setPreferableBackend(self.backendId)

    def setTarget(self, targetId):
        self.targetId = targetId
        self.net.setPreferableTarget(self.targetId)

    def preprocess(self, img):
        blob = np.transpose(img, (2, 0, 1))
        return blob[np.newaxis, :, :, :]

    def infer(self, srcimg):
        input_blob = self.preprocess(srcimg)

        self.net.setInput(input_blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        predictions = self.postprocess(outs[0])
        return predictions

    def postprocess(self, outputs):
        grids = []
        expanded_strides = []
        hsizes = [self.input_size[0] // stride for stride in self.strides]
        wsizes = [self.input_size[1] // stride for stride in self.strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, self.strides):
            xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        predictions = outputs[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.

        # multi-class nms
        final_dets = []
        for cls_ind in range(scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > self.confThreshold
            if valid_score_mask.sum() == 0:
                continue
            else:
                # call nms
                indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), cls_scores.tolist(), self.confThreshold, self.nmsThreshold)

                classids_ = np.ones((len(indices), 1)) * cls_ind
                final_dets.append(
                    np.concatenate([boxes_xyxy[indices], cls_scores[indices, None], classids_], axis=1)
                )

        if len(final_dets) == 0:
            return np.array([])

        return np.concatenate(final_dets, 0)
