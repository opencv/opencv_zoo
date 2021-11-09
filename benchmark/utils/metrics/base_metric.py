import cv2 as cv

from ..timer import Timer

class BaseMetric:
    def __init__(self, **kwargs):
        self._warmup = kwargs.pop('warmup', 3)
        self._repeat = kwargs.pop('repeat', 10)
        self._reduction = kwargs.pop('reduction', 'median')

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

    def _calcGMean(self, records, drop_largest=3):
        ''' Return the geometric mean of records after drop the first drop_largest
        '''
        l = len(records)
        if l <= drop_largest:
            print('len(records)({}) <= drop_largest({}), stop dropping.'.format(l, drop_largest))
        records_sorted = sorted(records, reverse=True)
        return sum(records_sorted[drop_largest:]) / (l - drop_largest)

    def _getResult(self):
        records = self._timer.getRecords()
        if self._reduction == 'median':
            return self._calcMedian(records)
        elif self._reduction == 'gmean':
            return self._calcGMean(records)
        else:
            raise NotImplementedError('Reduction {} is not supported'.format(self._reduction))

    def getReduction(self):
        return self._reduction

    def forward(self, model, *args, **kwargs):
        raise NotImplementedError('Not implemented')