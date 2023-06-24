import sys
import onnx
from onnxconverter_common import float16

op_block_list = ['ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper', 'DictVectorizer',
                 'FeatureVectorizer', 'Imputer', 'LabelEncoder', 'LinearClassifier', 'LinearRegressor',
                 'Normalizer', 'OneHotEncoder', 'RandomUniformLike', 'SVMClassifier', 'SVMRegressor', 'Scaler',
                 'TreeEnsembleClassifier', 'TreeEnsembleRegressor', 'ZipMap', 'NonMaxSuppression', 'TopK',
                 'RoiAlign', 'Range', 'CumSum', 'Min', 'Max', 'Upsample']


class Quantize:
    def __init__(self, model_path):
        self.model_path = model_path

    def run(self):
        model = onnx.load(self.model_path)
        model_fp16 = float16.convert_float_to_float16(model, op_block_list=op_block_list)
        output_name = '{}_fp16.onnx'.format(self.model_path[:-5])
        onnx.save(model_fp16, output_name)


models = dict(
    yunet=Quantize(model_path='../../models/face_detection_yunet/face_detection_yunet_2023mar.onnx'),
    sface=Quantize(model_path='../../models/face_recognition_sface/face_recognition_sface_2021dec.onnx'),
    fer=Quantize(model_path='../../models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx'),
    pphumanseg=Quantize(model_path='../../models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx'),
    mobilenetv1=Quantize(model_path='../../models/image_classification_mobilenet/image_classification_mobilenetv1_2022apr.onnx'),
    mobilenetv2=Quantize(model_path='../../models/image_classification_mobilenet/image_classification_mobilenetv2_2022apr.onnx'),
    ppresnet50=Quantize(model_path='../../models/image_classification_ppresnet/image_classification_ppresnet50_2022jan.onnx'),
    nanodet=Quantize(model_path='../../models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx'),
    yolox=Quantize(model_path='../../models/object_detection_yolox/object_detection_yolox_2022nov.onnx'),
    dasiamrpn=Quantize(model_path='../../models/object_tracking_dasiamrpn/object_tracking_dasiamrpn_model_2021nov.onnx'),
    dasiamrpn_cls1=Quantize(model_path='../../models/object_tracking_dasiamrpn/object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx'),
    dasiamrpn_r1=Quantize(model_path='../../models/object_tracking_dasiamrpn/object_tracking_dasiamrpn_kernel_r1_2021nov.onnx'),
    youtureid=Quantize(model_path='../../models/person_reid_youtureid/person_reid_youtu_2021nov.onnx'),
    mp_palmdet=Quantize(model_path='../../models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx'),
    mp_handpose=Quantize(model_path='../../models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx'),
    lpd_yunet=Quantize(model_path='../../models/license_plate_detection_yunet/license_plate_detection_lpd_yunet_2023mar.onnx'),
    mp_persondet=Quantize(model_path='../../models/person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx'),
    mp_pose=Quantize(model_path='../../models/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx'),
    db_en=Quantize(model_path='../../models/text_detection_db/text_detection_DB_IC15_resnet18_2021sep.onnx'),
    db_ch=Quantize(model_path='../../models/text_detection_db/text_detection_DB_TD500_resnet18_2021sep.onnx'),
    crnn_en=Quantize(model_path='../../models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx'),
    crnn_ch=Quantize(model_path='../../models/text_recognition_crnn/text_recognition_CRNN_CH_2021sep.onnx'),
    crnn_cn=Quantize(model_path='../../models/text_recognition_crnn/text_recognition_CRNN_CN_2021nov.onnx')
)

if __name__ == '__main__':
    selected_models = []
    for i in range(1, len(sys.argv)):
        selected_models.append(sys.argv[i])
    if not selected_models:
        selected_models = list(models.keys())
    print('Models to be quantized to fp16: {}'.format(str(selected_models)))

    for selected_model_name in selected_models:
        q = models[selected_model_name]
        print("------------------{}------------------".format(selected_model_name))
        q.run()
