from .base_dataloader import _BaseImageLoader, _BaseVideoLoader
from ..factory import DATALOADERS

@DATALOADERS.register
class BaseImageLoader(_BaseImageLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@DATALOADERS.register
class BaseVideoLoader(_BaseVideoLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)