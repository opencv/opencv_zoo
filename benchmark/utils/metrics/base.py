import cv2 as cv

from .base_metric import BaseMetric
from ..factory import METRICS

@METRICS.register
class Base(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, model, *args, **kwargs):
        img = args[0]
        if not self._sizes:
            h, w, _ = img.shape
            self._sizes.append([w, h])

        results = dict()
        self._timer.reset()
        for size in self._sizes:
            input_data = cv.resize(img, size)
            for _ in range(self._warmup):
                model.infer(input_data)
            for _ in range(self._repeat):
                self._timer.start()
                model.infer(input_data)
                self._timer.stop()
            results[str(size)] = self._getResult()

        return results