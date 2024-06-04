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
        
        self._outputNames = ['output_masks']  # actual output layer name
        self._currentInputSize = None
        self._inputSize = [640, 640]  # input size for the model

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
        
        # convert points to (640*640) size space
        for p in points:
            p[0] = int(p[0] * self._inputSize[0]/self._currentInputSize[0])
            p[1] = int(p[1]* self._inputSize[1]/self._currentInputSize[1])
             
        image_blob = cv.dnn.blobFromImage(image)
        
        points_blob = np.array([[points]], dtype=np.float32)
        
        labels_blob = np.array([[[labels]]])
        
        return image_blob, points_blob, labels_blob

    def infer(self, image, points, labels):
        # Preprocess
        imageBlob, pointsBlob, labelsBlob = self._preprocess(image, points, labels)
        # Forward
        self._model.setInput(imageBlob, self._inputNames[0])
        self._model.setInput(pointsBlob, self._inputNames[1])
        self._model.setInput(labelsBlob, self._inputNames[2])
        outputBlob = self._model.forward()
        # Postprocess
        results = self._postprocess(outputBlob)

        return results

    def _postprocess(self, outputBlob):
        mask = outputBlob[0, 0, 0, :, :] >= 0
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        # change to real image size
        mask_uint8 = cv.resize(mask_uint8, dsize=self._currentInputSize, interpolation=2)
                
        return mask_uint8
