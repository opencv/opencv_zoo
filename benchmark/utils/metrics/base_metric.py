import cv2 as cv

from ..timer import Timer

class BaseMetric:
    def __init__(self, **kwargs):
        self._warmup = kwargs.pop('warmup', 3)
        self._repeat = kwargs.pop('repeat', 10)

        self._timer = Timer()

    def _calcMedian(self, records):
        ''' Return the median of records
        '''
        l = len(records)
        mid = int(l / 2)
        if l % 2 == 0:
            return (records[mid] + records[mid - 1]) / 2
        else:
            return records[mid]

    def _calcMean(self, records, drop_largest=1):
        ''' Return the mean of records after dropping drop_largest
        '''
        l = len(records)
        if l <= drop_largest:
            print('len(records)({}) <= drop_largest({}), stop dropping.'.format(l, drop_largest))
        records_sorted = sorted(records, reverse=True)
        return sum(records_sorted[drop_largest:]) / (l - drop_largest)

    def _calcMin(self, records):
        return min(records)

    def getPerfStats(self, records):
        mean = self._calcMean(records, int(len(records) / 10))
        median = self._calcMedian(records)
        minimum = self._calcMin(records)
        return [mean, median, minimum]

    def forward(self, model, *args, **kwargs):
        raise NotImplementedError('Not implemented')
