import cv2 as cv

from .base_metric import BaseMetric
from ..factory import METRICS

@METRICS.register
class Recognition(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, model, *args, **kwargs):
        img, bboxes = args

        self._timer.reset()
        if bboxes is not None:
            for idx, bbox in enumerate(bboxes):
                for _ in range(self._warmup):
                    model.infer(img, bbox)
                for _ in range(self._repeat):
                    self._timer.start()
                    model.infer(img, bbox)
                    self._timer.stop()
        else:
            for _ in range(self._warmup):
                model.infer(img, None)
            for _ in range(self._repeat):
                self._timer.start()
                model.infer(img, None)
                self._timer.stop()

        return self._timer.getRecords()
