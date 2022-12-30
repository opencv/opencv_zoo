import numpy as np
import cv2 as cv

class MobileNetV2:
    def __init__(self, modelPath, labelPath=None, topK=1, backendId=0, targetId=0):
        self.model_path = modelPath
        self.label_path = labelPath
        assert topK >= 1
        self.top_k = topK
        self.backend_id = backendId
        self.target_id = targetId

        self.model = cv.dnn.readNet(self.model_path)
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

        self.input_names = ''
        self.output_names = ''
        self.input_size = [224, 224]
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        # load labels
        self._labels = self._load_labels()

    def _load_labels(self):
        labels = []
        if self.label_path is not None:
            with open(self.label_path, 'r') as f:
                for line in f:
                    labels.append(line.strip())
        return labels

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backendId):
        self.backend_id = backendId
        self.model.setPreferableBackend(self.backend_id)

    def setTarget(self, targetId):
        self.target_id = targetId
        self.model.setPreferableTarget(self.target_id)

    def _preprocess(self, image):
        input_blob = (image / 255.0 - self.mean) / self.std
        input_blob = input_blob.transpose(2, 0, 1)
        input_blob = input_blob[np.newaxis, :, :, :]
        input_blob = input_blob.astype(np.float32)
        return input_blob

    def infer(self, image):
        # Preprocess
        input_blob = self._preprocess(image)

        # Forward
        self.model.setInput(input_blob, self.input_names)
        output_blob = self.model.forward(self.output_names)

        # Postprocess
        results = self._postprocess(output_blob)

        return results

    def _postprocess(self, output_blob):
        batched_class_id_list = []
        for o in output_blob:
            class_id_list = o.argsort()[::-1][:self.top_k]
            batched_class_id_list.append(class_id_list)
        if len(self._labels) > 0:
            batched_predicted_labels = []
            for class_id_list in batched_class_id_list:
                predicted_labels = []
                for class_id in class_id_list:
                    predicted_labels.append(self._labels[class_id])
                batched_predicted_labels.append(predicted_labels)
            return batched_predicted_labels
        else:
            return batched_class_id_list

