# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.


import sys
import argparse

import numpy as np
import cv2 as cv

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

sys.path.append('../face_detection_yunet')
from yunet import YuNet

from ediffiqa import eDifFIQA

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

REFERENCE_FACIAL_POINTS = [
    [38.2946  , 51.6963  ],
    [73.5318  , 51.5014  ],
    [56.0252  , 71.7366  ],
    [41.5493  , 92.3655  ],
    [70.729904, 92.2041  ]
]

parser = argparse.ArgumentParser(description='eDifFIQA: Towards Efficient Face Image Quality Assessment based on Denoising Diffusion Probabilistic Models (https://github.com/LSIbabnikz/eDifFIQA).')
parser.add_argument('--input', '-i', type=str, default='./sample_image.jpg',
                    help='Usage: Set input to a certain image, defaults to "./sample_image.jpg".')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))

ediffiqa_parser = parser.add_argument_group("eDifFIQA", " Parameters of eDifFIQA - For face image quality assessment ")
ediffiqa_parser.add_argument('--model_q', '-mq', type=str, default='ediffiqa_tiny_jun2024.onnx',
                    help="Usage: Set model type, defaults to 'ediffiqa_tiny_jun2024.onnx'.")

yunet_parser = parser.add_argument_group("YuNet", " Parameters of YuNet - For face detection ")
yunet_parser.add_argument('--model_d', '-md', type=str, default='../face_detection_yunet/face_detection_yunet_2023mar.onnx',
                    help="Usage: Set model type, defaults to '../face_detection_yunet/face_detection_yunet_2023mar.onnx'.")
yunet_parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Usage: Set the minimum needed confidence for the model to identify a face, defauts to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
yunet_parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
yunet_parser.add_argument('--top_k', type=int, default=5000,
                    help='Usage: Keep top_k bounding boxes before NMS.')
args = parser.parse_args()


def visualize(image, results):
    output = image.copy()
    cv.putText(output, f"{results:.3f}", (0, 20), cv.FONT_HERSHEY_DUPLEX, .8, (0, 0, 255))

    return output


def align_image(image, detection_data):
    """ Performs face alignment on given image using the provided face landmarks (keypoints)

    Args:
        image (np.array): Unaligned face image
        detection_data (np.array): Detection data provided by YuNet 

    Returns:
        np.array: Aligned image
    """

    reference_pts = REFERENCE_FACIAL_POINTS

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    # Get source keypoints from YuNet detection data
    src_pts = np.float32(detection_data[0][4:-1]).reshape(5,2)
    src_pts_shp = src_pts.shape

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    tfm, _ = cv.estimateAffinePartial2D(src_pts, ref_pts, method=cv.LMEDS)

    face_img = cv.warpAffine(image, tfm, (112, 112))

    return face_img


if __name__ == '__main__':

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate eDifFIQA(T) (quality assesment)
    model_quality = eDifFIQA(
        modelPath=args.model_q,
        inputSize=[112, 112],
    )
    model_quality.setBackendAndTarget(
        backendId=backend_id,
        targetId=target_id
    )

    # Instantiate YuNet (face detection)
    model_detect = YuNet(
        modelPath=args.model_d,
        inputSize=[320, 320],
        confThreshold=args.conf_threshold,
        nmsThreshold=args.nms_threshold,
        topK=args.top_k,
        backendId=backend_id,
        targetId=target_id
    )

    # If input is an image
    image = cv.imread(args.input)
    h, w, _ = image.shape

    # Face Detection
    model_detect.setInputSize([w, h])
    results_detect = model_detect.infer(image)

    assert results_detect.size != 0, f" Face could not be detected in: {args.input}. "

    # Face Alignment
    aligned_image = align_image(image, results_detect)

    # Quality Assesment 
    quality = model_quality.infer(aligned_image)
    quality = np.squeeze(quality).item()

    viz_image = visualize(aligned_image, quality)

    print(f" Quality score of {args.input}: {quality:.3f} ")

    print(f" Saving visualization to results.jpg. ")
    cv.imwrite('results.jpg', viz_image)

