from .imagenet import ImageNet

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
