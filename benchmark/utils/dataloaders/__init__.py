from .base import BaseImageLoader, BaseVideoLoader
from .classification import ClassificationImageLoader
from .recognition import RecognitionImageLoader
from .tracking import TrackingVideoLoader

__all__ = ['BaseImageLoader', 'BaseVideoLoader', 'ClassificationImageLoader', 'RecognitionImageLoader', 'TrackingVideoLoader']