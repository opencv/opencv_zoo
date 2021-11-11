class Registery:
    def __init__(self, name):
        self._name = name
        self._dict = dict()

    def get(self, key):
        if key in self._dict:
            return self._dict[key]
        else:
            return self._dict['Base']

    def register(self, item):
        self._dict[item.__name__] = item
        # renaming *ImageLoader/*VideoLoader
        if 'ImageLoader' in item.__name__:
            name = item.__name__.replace('ImageLoader', '')
            self._dict[name] = item

METRICS = Registery('Metrics')
DATALOADERS = Registery('DataLoaders')