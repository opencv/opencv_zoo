import cv2 as cv

from .base_metric import BaseMetric
from ..factory import METRICS

@METRICS.register
class Estimation(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, model, *args, **kwargs):
        img, bbox, landmark = args

        self._timer.reset()
        for _ in range(self._warmup):
            model.infer(img, bbox, landmark)
        for _ in range(self._repeat):
            self._timer.start()
            model.infer(img, bbox, landmark)
            self._timer.stop()

        return self._getResult()