# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv # needs to have cv.wechat_qrcode_WeChatQRCode, which requires compile from source with opencv_contrib/modules/wechat_qrcode

class WeChatQRCode:
    def __init__(self, detect_prototxt_path, detect_model_path, sr_prototxt_path, sr_model_path, backendId=0, targetId=0):
        self._model = cv.wechat_qrcode_WeChatQRCode(
            detect_prototxt_path,
            detect_model_path,
            sr_prototxt_path,
            sr_model_path
        )
        if backendId != 0 and backendId != 3:
            raise NotImplementedError("Backend {} is not supported by cv.wechat_qrcode_WeChatQRCode()".format(backendId))
        if targetId != 0:
            raise NotImplementedError("Target {} is not supported by cv.wechat_qrcode_WeChatQRCode()")

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        if backendId != 0 and backendId != 3:
            raise NotImplementedError("Backend {} is not supported by cv.wechat_qrcode_WeChatQRCode()".format(backendId))
        if targetId != 0:
            raise NotImplementedError("Target {} is not supported by cv.wechat_qrcode_WeChatQRCode()")

    def infer(self, image):
        return self._model.detectAndDecode(image)
