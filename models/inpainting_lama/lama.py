import cv2 as cv
import numpy as np

class Lama:
    def __init__(self, modelPath='inpainting_lama_2025jan.onnx', backendId=0, targetId=0):
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

    def infer(self, image, mask):
        image_blob = cv.dnn.blobFromImage(image, 0.00392, (512, 512), (0,0,0), False, False)
        mask_blob = cv.dnn.blobFromImage(mask, scalefactor=1.0, size=(512, 512), mean=(0,), swapRB=False, crop=False)
        mask_blob = (mask_blob > 0).astype(np.float32)
        
        self._model.setInput(image_blob, "image")
        self._model.setInput(mask_blob, "mask")

        output = self._model.forward()

        # Postprocessing
        aspect_ratio = image.shape[0]/image.shape[1]
        result = output[0]
        result = np.transpose(result, (1, 2, 0))
        result = (result).astype(np.uint8)
        width = result.shape[1]
        height = int(width*aspect_ratio)
        result = cv.resize(result, (width, height))

        return result
