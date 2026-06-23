from itertools import product
import numpy as np
import cv2 as cv


class LPD_YuNet:
    def __init__(self, modelPath, inputSize=[320, 240], confThreshold=0.8,
                 nmsThreshold=0.3, topK=5000, keepTopK=750,
                 backendId=0, targetId=0):
        self.model_path = modelPath
        self.input_size = np.array(inputSize)
        self.confidence_threshold = confThreshold
        self.nms_threshold = nmsThreshold
        self.top_k = topK
        self.keep_top_k = keepTopK
        self.backend_id = backendId
        self.target_id = targetId

        self.output_names = ['loc', 'conf', 'iou']
        self.min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.steps = [8, 16, 32, 64]
        self.variance = [0.1, 0.2]

        # Load model
        self.model = cv.dnn.readNet(self.model_path)
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

        # Generate anchors/priorboxes
        self._priorGen()

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self.backend_id = backendId
        self.target_id = targetId
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

    def setInputSize(self, inputSize):
        self.input_size = inputSize
        self._priorGen()

    def _preprocess(self, image):
        return cv.dnn.blobFromImage(image)

    def infer(self, image):
        assert image.shape[0] == self.input_size[1], \
            f"{image.shape[0]} (height of input image) != {self.input_size[1]} (preset height)"
        assert image.shape[1] == self.input_size[0], \
            f"{image.shape[1]} (width of input image) != {self.input_size[0]} (preset width)"

        # Preprocess
        inputBlob = self._preprocess(image)

        # Forward
        self.model.setInput(inputBlob)
        outputBlob = self.model.forward(self.output_names)

        # Postprocess
        results = self._postprocess(outputBlob)
        return results

    def _postprocess(self, blob):
        # Decode outputs
        dets = self._decode(blob)

        # dets shape: [x1,y1,x2,y2,x3,y3,x4,y4,score]
        pts = dets[:, :-1].reshape(-1, 8)   # N x 8 corners
        scores = dets[:, -1].astype(float).tolist()

        # Convert corners → [x,y,w,h] for NMS
        bboxes = []
        for p in pts:
            xs = p[0::2]
            ys = p[1::2]
            x_min, y_min = float(xs.min()), float(ys.min())
            w, h = float(xs.max() - x_min), float(ys.max() - y_min)
            bboxes.append([x_min, y_min, w, h])

        keepIdx = cv.dnn.NMSBoxes(
            bboxes=bboxes,
            scores=scores,
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )

        # Normalize keepIdx across OpenCV versions
        if isinstance(keepIdx, tuple):
            keepIdx = keepIdx[0]
        if len(keepIdx) == 0:
            return np.empty((0, dets.shape[1]), dtype=dets.dtype)

        keepIdx = np.array(keepIdx).reshape(-1)

        # Keep original quadrilateral detections
        dets = dets[keepIdx]
        return dets[:self.keep_top_k]

    def _priorGen(self):
        w, h = self.input_size
        feature_map_2th = [int(int((h + 1) / 2) / 2),
                           int(int((w + 1) / 2) / 2)]
        feature_map_3th = [int(feature_map_2th[0] / 2),
                           int(feature_map_2th[1] / 2)]
        feature_map_4th = [int(feature_map_3th[0] / 2),
                           int(feature_map_3th[1] / 2)]
        feature_map_5th = [int(feature_map_4th[0] / 2),
                           int(feature_map_4th[1] / 2)]
        feature_map_6th = [int(feature_map_5th[0] / 2),
                           int(feature_map_5th[1] / 2)]

        feature_maps = [feature_map_3th, feature_map_4th,
                        feature_map_5th, feature_map_6th]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):  # i→h, j→w
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h
                    cx = (j + 0.5) * self.steps[k] / w
                    cy = (i + 0.5) * self.steps[k] / h
                    priors.append([cx, cy, s_kx, s_ky])
        self.priors = np.array(priors, dtype=np.float32)

    def _decode(self, blob):
        loc, conf, iou = blob

        # get score
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]

        # clamp
        iou_scores = np.clip(iou_scores, 0., 1.)
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = self.input_size

        # get four corner points
        bboxes = np.hstack((
            (self.priors[:, 0:2] + loc[:, 4:6] * self.variance[0] * self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2] + loc[:, 6:8] * self.variance[0] * self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2] + loc[:, 10:12] * self.variance[0] * self.priors[:, 2:4]) * scale,
            (self.priors[:, 0:2] + loc[:, 12:14] * self.variance[0] * self.priors[:, 2:4]) * scale
        ))

        dets = np.hstack((bboxes, scores))
        return dets
