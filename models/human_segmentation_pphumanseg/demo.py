# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import numpy as np
import cv2 as cv

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from pphumanseg import PPHumanSeg

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='PPHumanSeg (https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/contrib/PP-HumanSeg)')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input path to a certain image, omit if using camera.')
parser.add_argument('--model', '-m', type=str, default='human_segmentation_pphumanseg_2023mar.onnx',
                    help='Usage: Set model path, defaults to human_segmentation_pphumanseg_2023mar.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save a file with results. Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map

def visualize(image, result, weight=0.6, fps=None):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The input image.
        result (np.ndarray): The predict result of image.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6
        fps (str): The FPS to be drawn on the input image.

    Returns:
        vis_result (np.ndarray): The visualized result.
    """
    color_map = get_color_map_list(256)
    color_map = np.array(color_map).reshape(256, 3).astype(np.uint8)

    # Use OpenCV LUT for color mapping
    c1 = cv.LUT(result, color_map[:, 0])
    c2 = cv.LUT(result, color_map[:, 1])
    c3 = cv.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))

    vis_result = cv.addWeighted(image, weight, pseudo_img, 1 - weight, 0)

    if fps is not None:
        cv.putText(vis_result, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    return vis_result


if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    # Instantiate PPHumanSeg
    model = PPHumanSeg(modelPath=args.model, backendId=backend_id, targetId=target_id)

    if args.input is not None:
        # Read image and resize to 192x192
        image = cv.imread(args.input)
        h, w, _ = image.shape
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        _image = cv.resize(image, dsize=(192, 192))

        # Inference
        result = model.infer(_image)
        result = cv.resize(result[0, :, :], dsize=(w, h), interpolation=cv.INTER_NEAREST)

        # Draw results on the input image
        image = visualize(image, result)

        # Save results if save is true
        if args.save:
            print('Results saved to result.jpg\n')
            cv.imwrite('result.jpg', image)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, image)
            cv.waitKey(0)
    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            _frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            _frame = cv.resize(_frame, dsize=(192, 192))

            # Inference
            tm.start()
            result = model.infer(_frame)
            tm.stop()
            result = cv.resize(result[0, :, :], dsize=(w, h), interpolation=cv.INTER_NEAREST)

            # Draw results on the input image
            frame = visualize(frame, result, fps=tm.getFPS())

            # Visualize results in a new window
            cv.imshow('PPHumanSeg Demo', frame)

            tm.reset()

