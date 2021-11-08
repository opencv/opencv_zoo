import cv2 as cv

class Timer:
    def __init__(self):
        self._tm = cv.TickMeter()
        self._record = []

    def start(self):
        self._tm.start()

    def stop(self):
        self._tm.stop()
        self._record.append(self._tm.getTimeMilli())
        self._tm.reset()

    def reset(self):
        self._record = []

    def getRecords(self):
        return self._record