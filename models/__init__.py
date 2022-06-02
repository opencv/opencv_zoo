from .face_detection_yunet.yunet import YuNet
from .text_detection_db.db import DB
from .text_recognition_crnn.crnn import CRNN
from .face_recognition_sface.sface import SFace
from .image_classification_ppresnet.ppresnet import PPResNet
from .human_segmentation_pphumanseg.pphumanseg import PPHumanSeg
from .qrcode_wechatqrcode.wechatqrcode import WeChatQRCode
from .object_tracking_dasiamrpn.dasiamrpn import DaSiamRPN
from .person_reid_youtureid.youtureid import YoutuReID
from .image_classification_mobilenet.mobilenet_v1 import MobileNetV1
from .image_classification_mobilenet.mobilenet_v2 import MobileNetV2
from .palm_detection_mediapipe.mp_palmdet import MPPalmDet
from .license_plate_detection_yunet.lpd_yunet import LPD_YuNet

class Registery:
    def __init__(self, name):
        self._name = name
        self._dict = dict()

    def get(self, key):
        return self._dict[key]

    def register(self, item):
        self._dict[item.__name__] = item

MODELS = Registery('Models')
MODELS.register(YuNet)
MODELS.register(DB)
MODELS.register(CRNN)
MODELS.register(SFace)
MODELS.register(PPResNet)
MODELS.register(PPHumanSeg)
MODELS.register(WeChatQRCode)
MODELS.register(DaSiamRPN)
MODELS.register(YoutuReID)
MODELS.register(MobileNetV1)
MODELS.register(MobileNetV2)
MODELS.register(MPPalmDet)
MODELS.register(LPD_YuNet)
