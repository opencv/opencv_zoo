import cv2 as cv
import numpy as np

class Dexined:
    def __init__(self, modelPath='edge_detection_dexined_2024sep.onnx', backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId
        
        # Load the model
        self._model = cv.dnn.readNetFromONNX(self._modelPath)
        self.setBackendAndTarget(self._backendId, self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def postProcessing(self, output, shape):
        h, w = shape
        preds = []
        for p in output:
            img = self.sigmoid(p)
            img = np.squeeze(img)
            img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            img = cv.resize(img, (w, h))
            preds.append(img)
        fuse = preds[-1]
        ave = np.array(preds, dtype=np.float32)
        ave = np.uint8(np.mean(ave, axis=0))
        return fuse, ave

    def infer(self, image):
        inp = cv.dnn.blobFromImage(image, 1.0, (512, 512), (103.5, 116.2, 123.6), swapRB=False, crop=False)
        self._model.setInput(inp)
        
        # Forward pass through the model
        out = self._model.forward()
        result, _ = self.postProcessing(out, image.shape[:2])

        return result
