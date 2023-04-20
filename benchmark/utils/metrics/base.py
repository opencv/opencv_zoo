import cv2 as cv

from .base_metric import BaseMetric
from ..factory import METRICS

@METRICS.register
class Base(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, model, *args, **kwargs):
        img = args[0]

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
