# This file is part of OpenCV Zoo project.

import cv2 as cv
import numpy as np
import onnxruntime


class Raft:
    def __init__(self, modelPath):
        self._modelPath = modelPath
        self.session = onnxruntime.InferenceSession(self._modelPath, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.output_shape = model_outputs[0].shape
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]

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
        output = self.session.run(self.output_names, {self.input_names[0]: input_1, 
													   self.input_names[1]: input_2})
        # Postprocess
        results = self._postprocess(output)

        return results

    def _postprocess(self, output):
        
        flow_map = output[1][0].transpose(1, 2, 0)
        return flow_map