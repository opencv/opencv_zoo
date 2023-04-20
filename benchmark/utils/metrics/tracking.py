import cv2 as cv

from .base_metric import BaseMetric
from ..factory import METRICS

@METRICS.register
class Tracking(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # if self._warmup or self._repeat:
        #     print('warmup and repeat in metric for tracking do not function.')

    def forward(self, model, *args, **kwargs):
        stream, first_frame, rois = args

        for roi in rois:
            stream.reload()
            model.init(first_frame, tuple(roi))
            self._timer.reset()
            for frame in stream:
                self._timer.start()
                model.infer(frame)
                self._timer.stop()

        return self._timer.getRecords()
