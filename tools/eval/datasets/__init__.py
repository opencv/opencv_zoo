from .imagenet import ImageNet
from .widerface import WIDERFace
from .lfw import LFW
from .icdar import ICDAR
from .iiit5k import IIIT5K
from .minisupervisely import MiniSupervisely

class Registery:
    def __init__(self, name):
        self._name = name
        self._dict = dict()

    def get(self, key):
        return self._dict[key]

    def register(self, item):
        self._dict[item.__name__] = item

DATASETS = Registery("Datasets")
DATASETS.register(ImageNet)
DATASETS.register(WIDERFace)
DATASETS.register(LFW)
DATASETS.register(ICDAR)
DATASETS.register(IIIT5K)
DATASETS.register(MiniSupervisely)
