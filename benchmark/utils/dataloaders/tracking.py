import os
import numpy as np

from .base_dataloader import _BaseVideoLoader
from ..factory import DATALOADERS

@DATALOADERS.register
class TrackingVideoLoader(_BaseVideoLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._first_frames = dict()
        for filename in self._files:
            stream = self._streams[filename]
            self._first_frames[filename] = next(stream)

        self._rois = self._load_roi()

    def _load_roi(self):
        rois = dict.fromkeys(self._files, None)
        for filename in self._files:
            rois[filename] = np.loadtxt(os.path.join(self._path, '{}.txt'.format(filename[:-4])), dtype=np.int32, ndmin=2)
        return rois

    def __getitem__(self, idx):
        filename = self._files[idx]
        return filename, self._streams[filename], self._first_frames[filename], self._rois[filename]