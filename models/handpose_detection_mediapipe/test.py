import numpy as np
import cv2 as cv


net = cv.dnn.readNet('./handpose_detection_mphandpose_2022may.onnx')

input_blob = np.random.randn(1, 256, 256, 3)
net.setInput(input_blob)

net.forward()
