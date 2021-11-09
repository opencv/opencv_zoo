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
        if item.__name__ == 'BaseImageLoader':
            self._dict['Base'] = item

METRICS = Registery('Metrics')
DATALOADERS = Registery('DataLoaders')