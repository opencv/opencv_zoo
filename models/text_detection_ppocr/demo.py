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

from ppocr_det import PPOCRDet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='PP-OCR Text Detection (https://arxiv.org/abs/2206.03001).')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='./text_detection_en_ppocrv3_2023may.onnx',
                    help='Usage: Set model path, defaults to text_detection_en_ppocrv3_2023may.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--width', type=int, default=736,
                    help='Usage: Resize input image to certain width, default = 736. It should be multiple by 32.')
parser.add_argument('--height', type=int, default=736,
                    help='Usage: Resize input image to certain height, default = 736. It should be multiple by 32.')
parser.add_argument('--binary_threshold', type=float, default=0.3,
                    help='Usage: Threshold of the binary map, default = 0.3.')
parser.add_argument('--polygon_threshold', type=float, default=0.5,
                    help='Usage: Threshold of polygons, default = 0.5.')
parser.add_argument('--max_candidates', type=int, default=200,
                    help='Usage: Set maximum number of polygon candidates, default = 200.')
parser.add_argument('--unclip_ratio', type=np.float64, default=2.0,
                    help=' Usage: The unclip ratio of the detected text region, which determines the output size, default = 2.0.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), isClosed=True, thickness=2, fps=None):
    output = image.copy()

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    pts = np.array(results[0])
    output = cv.polylines(output, pts, isClosed, box_color, thickness)

    return output

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate model
    model = PPOCRDet(modelPath=args.model,
               inputSize=[args.width, args.height],
               binaryThreshold=args.binary_threshold,
               polygonThreshold=args.polygon_threshold,
               maxCandidates=args.max_candidates,
               unclipRatio=args.unclip_ratio,
               backendId=backend_id,
               targetId=target_id)

    # If input is an image
    if args.input is not None:
        original_image = cv.imread(args.input)
        original_w = original_image.shape[1]
        original_h = original_image.shape[0]
        scaleHeight = original_h / args.height
        scaleWidth = original_w / args.width
        image = cv.resize(original_image, [args.width, args.height])

        # Inference
        results = model.infer(image)

        # Scale the results bounding box
        for i in range(len(results[0])):
            for j in range(4):
                box = results[0][i][j]
                results[0][i][j][0] = box[0] * scaleWidth
                results[0][i][j][1] = box[1] * scaleHeight

        # Print results
        print('{} texts detected.'.format(len(results[0])))
        for idx, (bbox, score) in enumerate(zip(results[0], results[1])):
            print('{}: {} {} {} {}, {:.2f}'.format(idx, bbox[0], bbox[1], bbox[2], bbox[3], score))

        # Draw results on the input image
        original_image = visualize(original_image, results)

        # Save results if save is true
        if args.save:
            print('Resutls saved to result.jpg\n')
            cv.imwrite('result.jpg', original_image)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, original_image)
            cv.waitKey(0)
    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, original_image = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            original_w = original_image.shape[1]
            original_h = original_image.shape[0]
            scaleHeight = original_h / args.height
            scaleWidth = original_w / args.width
            frame = cv.resize(original_image, [args.width, args.height])
            # Inference
            tm.start()
            results = model.infer(frame) # results is a tuple
            tm.stop()

            # Scale the results bounding box
            for i in range(len(results[0])):
                for j in range(4):
                    box = results[0][i][j]
                    results[0][i][j][0] = box[0] * scaleWidth
                    results[0][i][j][1] = box[1] * scaleHeight

            # Draw results on the input image
            original_image = visualize(original_image, results, fps=tm.getFPS())

            # Visualize results in a new Window
            cv.imshow('{} Demo'.format(model.name), original_image)

            tm.reset()
