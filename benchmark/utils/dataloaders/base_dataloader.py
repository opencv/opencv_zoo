import os

import cv2 as cv

class _BaseImageLoader:
    def __init__(self, **kwargs):
        self._path = kwargs.pop('path', None)
        assert self._path, 'Benchmark[\'data\'][\'path\'] cannot be empty.'

        self._files = kwargs.pop('files', None)
        assert self._files, 'Benchmark[\'data\'][\'files\'] cannot be empty'
        self._len_files = len(self._files)

        self._sizes = kwargs.pop('sizes', [[0, 0]])
        self._len_sizes = len(self._sizes)

    @property
    def name(self):
        return self.__class__.__name__

    def __len__(self):
        return self._len_files * self._len_sizes

    def __iter__(self):
        for filename in self._files:
            image = cv.imread(os.path.join(self._path, filename))
            if [0, 0] in self._sizes:
                yield filename, image
            else:
                for size in self._sizes:
                    image_r = cv.resize(image, size)
                    yield filename, image_r

class _VideoStream:
    def __init__(self, filepath):
        self._filepath = filepath
        self._video = cv.VideoCapture(self._filepath)

    def __iter__(self):
        while True:
            has_frame, frame = self._video.read()
            if has_frame:
                yield frame
            else:
                break

    def __next__(self):
        while True:
            has_frame, frame = self._video.read()
            if has_frame:
                return frame
            else:
                break

    def reload(self):
        self._video = cv.VideoCapture(self._filepath)

    def getFrameSize(self):
        w = int(self._video.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(self._video.get(cv.CAP_PROP_FRAME_HEIGHT))
        return [w, h]


class _BaseVideoLoader:
    def __init__(self, **kwargs):
        self._path = kwargs.pop('path', None)
        assert self._path, 'Benchmark[\'data\'][\'path\'] cannot be empty.'

        self._files = kwargs.pop('files', None)
        assert self._files,'Benchmark[\'data\'][\'files\'] cannot be empty.'

        self._streams = dict()
        for filename in self._files:
            self._streams[filename] = _VideoStream(os.path.join(self._path, filename))

    @property
    def name(self):
        return self.__class__.__name__

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        return self._files[idx], self._streams[idx]