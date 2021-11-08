import cv2 as cv

from .base_metric import BaseMetric
from ..factory import METRICS

@METRICS.register
class Recognition(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, model, *args, **kwargs):
        img, bboxes = args
        if not self._sizes:
            h, w, _ = img.shape
            self._sizes.append([w, h])

        results = dict()
        self._timer.reset()
        for idx, bbox in enumerate(bboxes):
            for _ in range(self._warmup):
                model.infer(img, bbox)
            for _ in range(self._repeat):
                self._timer.start()
                model.infer(img, bbox)
                self._timer.stop()
            results['bbox{}'.format(idx)] = self._getResult()

        return results