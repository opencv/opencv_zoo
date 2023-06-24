import os
import sys
import argparse

import numpy as np
import cv2 as cv

from datasets import DATASETS

if "PYTHONPATH" in os.environ:
    root_dir = os.environ["PYTHONPATH"]
else:
    root_dir = os.path.join("..", "..")
sys.path.append(root_dir)
from models import MODELS

parser = argparse.ArgumentParser("Evaluation with OpenCV on different models in the zoo.")
parser.add_argument("--model", "-m", type=str, required=True, help="model name")
parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name")
parser.add_argument("--dataset_root", "-dr", type=str, required=True, help="Root directory of given dataset")
args = parser.parse_args()

models = dict(
    mobilenetv1=dict(
        name="MobileNet",
        topic="image_classification",
        modelPath=os.path.join(root_dir, "models/image_classification_mobilenet/image_classification_mobilenetv1_2022apr.onnx"),
        topK=5,
        loadLabel=False),
    mobilenetv1_q=dict(
        name="MobileNet",
        topic="image_classification",
        modelPath=os.path.join(root_dir, "models/image_classification_mobilenet/image_classification_mobilenetv1_2022apr_int8.onnx"),
        topK=5,
        loadLabel=False),
    mobilenetv2=dict(
        name="MobileNet",
        topic="image_classification",
        modelPath=os.path.join(root_dir, "models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx"),
        topK=5,
        loadLabel=False),
    mobilenetv2_q=dict(
        name="MobileNet",
        topic="image_classification",
        modelPath=os.path.join(root_dir, "models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr_int8.onnx"),
        topK=5,
        loadLabel=False),
    ppresnet=dict(
        name="PPResNet",
        topic="image_classification",
        modelPath=os.path.join(root_dir, "models/image_classification_ppresnet/image_classification_ppresnet50_2022jan.onnx"),
        topK=5,
        loadLabel=False),
    ppresnet_q=dict(
        name="PPResNet",
        topic="image_classification",
        modelPath=os.path.join(root_dir, "models/image_classification_ppresnet/image_classification_ppresnet50_2022jan_int8.onnx"),
        topK=5,
        loadLabel=False),
    yunet=dict(
        name="YuNet",
        topic="face_detection",
        modelPath=os.path.join(root_dir, "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
        topK=5000,
        confThreshold=0.3,
        nmsThreshold=0.45),
    yunet_q=dict(
        name="YuNet",
        topic="face_detection",
        modelPath=os.path.join(root_dir, "models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx"),
        topK=5000,
        confThreshold=0.3,
        nmsThreshold=0.45),
    sface=dict(
        name="SFace",
        topic="face_recognition",
        modelPath=os.path.join(root_dir, "models/face_recognition_sface/face_recognition_sface_2021dec.onnx")),
    sface_q=dict(
        name="SFace",
        topic="face_recognition",
        modelPath=os.path.join(root_dir, "models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx")),
    crnn_en=dict(
        name="CRNN",
        topic="text_recognition",
        modelPath=os.path.join(root_dir, "models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx")),
    crnn_en_q=dict(
        name="CRNN",
        topic="text_recognition",
        modelPath=os.path.join(root_dir, "models/text_recognition_crnn/text_recognition_CRNN_EN_2022oct_int8.onnx")),
    pphumanseg=dict(
        name="PPHumanSeg",
        topic="human_segmentation",
        modelPath=os.path.join(root_dir, "models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx")),
    pphumanseg_q=dict(
        name="PPHumanSeg",
        topic="human_segmentation",
        modelPath=os.path.join(root_dir, "models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar_int8.onnx")),
)

datasets = dict(
        imagenet=dict(
            name="ImageNet",
            topic="image_classification",
            size=224),
        widerface=dict(
            name="WIDERFace",
            topic="face_detection"),
        lfw=dict(
            name="LFW",
            topic="face_recognition",
            target_size=112),
        icdar=dict(
            name="ICDAR",
            topic="text_recognition"),
        iiit5k=dict(
            name="IIIT5K",
            topic="text_recognition"),
        mini_supervisely=dict(
            name="MiniSupervisely",
            topic="human_segmentation"),
)

def main(args):
    # Instantiate model
    model_key = args.model.lower()
    assert model_key in models

    model_name = models[model_key].pop("name")
    model_topic = models[model_key].pop("topic")
    model_handler, _ = MODELS.get(model_name)
    model = model_handler(**models[model_key])

    # Instantiate dataset
    dataset_key = args.dataset.lower()
    assert dataset_key in datasets

    dataset_name = datasets[dataset_key].pop("name")
    dataset_topic = datasets[dataset_key].pop("topic")
    dataset = DATASETS.get(dataset_name)(root=args.dataset_root, **datasets[dataset_key])

    # Check if model_topic matches dataset_topic
    assert model_topic == dataset_topic

    # Run evaluation
    dataset.eval(model)
    dataset.print_result()

if __name__ == "__main__":
    main(args)
