import cv2 as cv
import numpy as np

class Nafnet:
    def __init__(self, modelPath='deblurring_nafnet_2025may.onnx', backendId=0, targetId=0):
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

    def infer(self, image):
        image_blob = cv.dnn.blobFromImage(image, 0.00392, (image.shape[1], image.shape[0]), (0,0,0), True, False)

        self._model.setInput(image_blob)
        output = self._model.forward()

        # Postprocessing
        result = output[0]
        result = np.transpose(result, (1, 2, 0))
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result = cv.cvtColor(result, cv.COLOR_RGB2BGR)

        return result
