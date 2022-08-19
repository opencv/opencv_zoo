import cv2
import numpy as np

class YoloX():
    def __init__(self, modelPath, p6=False, confThreshold=0.35, nmsThreshold=0.5, objThreshold=0.5):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.num_classes = len(self.classes)
        self.net = cv2.dnn.readNet(modelPath)
        self.input_size = (640, 640)
        #self.ratio = min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        if not p6:
            self.strides = [8, 16, 32]
        else:
            self.strides = [8, 16, 32, 64]
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def preprocess(self, image):
        if len(image.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(self.input_size) * 114.0
        img = np.array(image)
        ratio = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img
        image = padded_img

        image = image.astype(np.float32)
        image = image[:, :, ::-1]
        return image, ratio

    def infer(self, srcimg):
        img, ratio = self.preprocess(srcimg)
        blob = cv2.dnn.blobFromImage(img)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        predictions = self.postprocess(outs[0], ratio)

        return predictions


    def postprocess(self, outputs, ratio):
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
        boxes_xyxy /= ratio

        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > self.confThreshold

            if valid_score_mask.sum() == 0:
                continue

            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes_xyxy[valid_score_mask]

                keep = []
                x1 = valid_boxes[:, 0]
                y1 = valid_boxes[:, 1]
                x2 = valid_boxes[:, 2]
                y2 = valid_boxes[:, 3]

                areas = (x2 - x1 + 1) * (y2 - y1 + 1)
                order = valid_scores.argsort()[::-1]

                while order.size > 0:
                    i = order[0]
                    keep.append(i)
                    xx1 = np.maximum(x1[i], x1[order[1:]])
                    yy1 = np.maximum(y1[i], y1[order[1:]])
                    xx2 = np.minimum(x2[i], x2[order[1:]])
                    yy2 = np.minimum(y2[i], y2[order[1:]])

                    w = np.maximum(0.0, xx2 - xx1 + 1)
                    h = np.maximum(0.0, yy2 - yy1 + 1)
                    inter = w * h
                    ovr = inter / (areas[i] + areas[order[1:]] - inter)

                    inds = np.where(ovr <= self.nmsThreshold)[0]
                    order = order[inds + 1]
                    if len(keep) > 0:
                        cls_inds = np.ones((len(keep), 1)) * cls_ind
                        dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                        final_dets.append(dets)

        if len(final_dets) == 0:
            return None

        res_dets = np.concatenate(final_dets, 0)
        return res_dets
