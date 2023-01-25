from pathlib import Path
import glob
import os

from .face_detection_yunet.yunet import YuNet
from .text_detection_db.db import DB
from .text_recognition_crnn.crnn import CRNN
from .face_recognition_sface.sface import SFace
from .image_classification_ppresnet.ppresnet import PPResNet
from .human_segmentation_pphumanseg.pphumanseg import PPHumanSeg
from .qrcode_wechatqrcode.wechatqrcode import WeChatQRCode
from .object_tracking_dasiamrpn.dasiamrpn import DaSiamRPN
from .person_reid_youtureid.youtureid import YoutuReID
from .image_classification_mobilenet.mobilenet import MobileNet
from .palm_detection_mediapipe.mp_palmdet import MPPalmDet
from .handpose_estimation_mediapipe.mp_handpose import MPHandPose
from .license_plate_detection_yunet.lpd_yunet import LPD_YuNet
from .object_detection_nanodet.nanodet import NanoDet
from .object_detection_yolox.yolox import YoloX
from .facial_expression_recognition.facial_fer_model import FacialExpressionRecog

class ModuleRegistery:
    def __init__(self, name):
        self._name = name
        self._dict = dict()

        self._base_path = Path(__file__).parent

    def get(self, key):
        '''
        Returns a tuple with:
        - a module handler,
        - a list of model file paths
        '''
        return self._dict[key]

    def register(self, item):
        '''
        Registers given module handler along with paths of model files
        '''
        # search for model files
        model_dir = str(self._base_path / item.__module__.split(".")[1])
        fp32_model_paths = []
        fp16_model_paths = []
        int8_model_paths = []
        # onnx
        ret_onnx = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
        if "object_tracking" in item.__module__:
            # object tracking models usually have multiple parts
            fp32_model_paths = [ret_onnx]
        else:
            for r in ret_onnx:
                if "int8" in r:
                    int8_model_paths.append([r])
                elif "fp16" in r: # exclude fp16 for now
                    fp16_model_paths.append([r])
                else:
                    fp32_model_paths.append([r])
        # caffe
        ret_caffemodel = sorted(glob.glob(os.path.join(model_dir, "*.caffemodel")))
        ret_prototxt = sorted(glob.glob(os.path.join(model_dir, "*.prototxt")))
        caffe_models = []
        for caffemodel, prototxt in zip(ret_caffemodel, ret_prototxt):
            caffe_models += [prototxt, caffemodel]
        if caffe_models:
            fp32_model_paths.append(caffe_models)

        all_model_paths = dict(
            fp32=fp32_model_paths,
            fp16=fp16_model_paths,
            int8=int8_model_paths,
        )

        self._dict[item.__name__] = (item, all_model_paths)

MODELS = ModuleRegistery('Models')
MODELS.register(YuNet)
MODELS.register(DB)
MODELS.register(CRNN)
MODELS.register(SFace)
MODELS.register(PPResNet)
MODELS.register(PPHumanSeg)
MODELS.register(WeChatQRCode)
MODELS.register(DaSiamRPN)
MODELS.register(YoutuReID)
MODELS.register(MobileNet)
MODELS.register(MPPalmDet)
MODELS.register(MPHandPose)
MODELS.register(LPD_YuNet)
MODELS.register(NanoDet)
MODELS.register(YoloX)
MODELS.register(FacialExpressionRecog)
