import cv2
import onnx
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box

onnx_name = 'w6-pose.onnx'
model_ = onnx.load(onnx_name)
onnx.checker.check_model(model_)
model = cv2.dnn.readNetFromONNX(onnx_name)
print("loading complete")


# 读取图片并预处理
image = cv2.imread('./test_data/2.jpg')
image = letterbox(image, 960, stride=64, auto=True)[0]
# print(image.shape)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (640, 640)))
# print(blob.shape)

# 推理
model.setInput(blob)
output = model.forward()
print(output.shape)
