import numpy as np
import cv2 as cv

class Timer:
    def __init__(self, warmup=0, reduction='median'):
        self._warmup = warmup
        self._reduction = reduction
        self._tm = cv.TickMeter()
        self._time_record = []
        self._calls = 0

    def start(self):
        self._tm.start()

    def stop(self):
        self._tm.stop()
        self._calls += 1
        self._time_record.append(self._tm.getTimeMilli())
        self._tm.reset()

    def reset(self):
        self._time_record = []
        self._calls = 0

    def getResult(self):
        if self._reduction == 'median':
            return self._getMedian(self._time_record[self._warmup:])
        elif self._reduction == 'gmean':
            return self._getGMean(self._time_record[self._warmup:])
        else:
            raise NotImplementedError()

    def _getMedian(self, records):
        ''' Return median time
        '''
        l = len(records)
        mid = int(l / 2)
        if l % 2 == 0:
            return (records[mid] + records[mid - 1]) / 2
        else:
            return records[mid]

    def _getGMean(self, records, drop_largest=3):
        ''' Return geometric mean of time
        '''
        time_record_sorted = sorted(records, reverse=True)
        return sum(records[drop_largest:]) / (self._calls - drop_largest)

class Metric:
    def __init__(self, **kwargs):
        self._sizes = kwargs.pop('sizes', None)
        self._warmup = kwargs.pop('warmup', 3)
        self._repeat = kwargs.pop('repeat', 10)
        assert self._warmup < self._repeat, 'The value of warmup must be smaller than the value of repeat.'
        self._batch_size = kwargs.pop('batchSize', 1)
        self._reduction = kwargs.pop('reduction', 'median')

        self._timer = Timer(self._warmup, self._reduction)

    def getReduction(self):
        return self._reduction

    def forward(self, model, *args, **kwargs):
        img = args[0]
        h, w, _ = img.shape
        if not self._sizes:
            self._sizes = [[w, h]]

        results = dict()
        self._timer.reset()
        if len(args) == 1:
            for size in self._sizes:
                img_r = cv.resize(img, size)
                try:
                    model.setInputSize(size)
                except:
                    pass
                # TODO: batched inference
                # input_data = [img] * self._batch_size
                input_data = img_r
                for _ in range(self._repeat+self._warmup):
                    self._timer.start()
                    model.infer(input_data)
                    self._timer.stop()
                results[str(size)] = self._timer.getResult()
        else:
            # TODO: batched inference
            # input_data = [args] * self._batch_size
            bboxes = args[1]
            for idx, bbox in enumerate(bboxes):
                for _ in range(self._repeat+self._warmup):
                    self._timer.start()
                    model.infer(img, bbox)
                    self._timer.stop()
                results['bbox{}'.format(idx)] = self._timer.getResult()

        return results