import numpy as np

from .base_dataloader import _BaseVideoLoader
from ..factory import DATALOADERS

@DATALOADERS.register
class TrackingVideoLoader(_BaseVideoLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._rois = self._load_roi()

        unsupported_keys = []
        for k, _ in kwargs.items():
            unsupported_keys.append(k)
        print('Keys ({}) are not supported in Benchmark[\'data\'].'.format(str(unsupported_keys)))

    def _load_roi(self):
        rois = dict.fromkeys(self._files, None)
        for filename in self._files:
            rois[filename] = np.loadtxt(os.path.join(self._path, '{}.txt'.format(filename[:-4])), ndmin=2)
        return rois

    def getROI(self):
        return self._rois