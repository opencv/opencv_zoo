# This file is part of OpenCV Zoo project.

import cv2 as cv
import numpy as np


class Raft:
    def __init__(self, modelPath):
        self._modelPath = modelPath
        self.model = cv.dnn.readNet(self._modelPath)
        
        self.input_names = ['0', '1']
        self.first_input_name = self.input_names[0]
        self.second_input_name = self.input_names[1]
        self.input_shape = [360, 480] # change if going to use different model with different input shape
        self.input_height = self.input_shape[0]
        self.input_width = self.input_shape[1]

    @property
    def name(self):
        return self.__class__.__name__

    def _preprocess(self, image):
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img_input = cv.resize(image, (self.input_width,self.input_height))
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis,:,:,:]
        img_input = img_input.astype(np.float32)
        return img_input

    def infer(self, image1, image2):

        # Preprocess
        input_1 = self._preprocess(image1)
        input_2 = self._preprocess(image2)
        
        # Forward
        self.model.setInput(input_1, self.first_input_name)
        self.model.setInput(input_2, self.second_input_name)
        layer_names = self.model.getLayerNames()
        outputlayers = [layer_names[i-1] for i in self.model.getUnconnectedOutLayers()]
        output = self.model.forward(outputlayers)
        
        # Postprocess
        results = self._postprocess(output)

        return results

    def _postprocess(self, output):
        
        flow_map = output[1][0].transpose(1, 2, 0)
        return flow_map