import cv2 as cv

from .base_metric import BaseMetric
from ..factory import METRICS

@METRICS.register
class Detection(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, model, *args, **kwargs):
        img = args[0]
        size = [img.shape[1], img.shape[0]]
        try:
            model.setInputSize(size)
        except:
            pass

        # warmup
        for _ in range(self._warmup):
            model.infer(img)
        # repeat
        self._timer.reset()
        for _ in range(self._repeat):
            self._timer.start()
            model.infer(img)
            self._timer.stop()

        return self._timer.getRecords()
