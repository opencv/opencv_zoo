import argparse

import numpy as np
import cv2 as cv

from mp_persondet import MPPersonDet

# Check OpenCV version
assert cv.__version__ >= "4.9.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='Person Detector from MediaPipe')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='./person_detection_mediapipe_2023mar.onnx',
                    help='Usage: Set model path, defaults to person_detection_mediapipe_2023mar.onnx')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--score_threshold', type=float, default=0.5,
                    help='Usage:  Set the minimum needed confidence for the model to identify a person, defaults to 0.5. Smaller values may result in faster detection, but will limit accuracy. Filter out persons of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
parser.add_argument('--top_k', type=int, default=1,
                    help='Usage: Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

def visualize(image, results, fps=None):
    output = image.copy()

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    for idx, person in enumerate(results):
        score = person[-1]
        person_landmarks = person[4:-1].reshape(4, 2).astype(np.int32)

        hip_point = person_landmarks[0]
        full_body = person_landmarks[1]
        shoulder_point = person_landmarks[2]
        upper_body = person_landmarks[3]

        # draw circle for full body
        radius = np.linalg.norm(hip_point - full_body).astype(np.int32)
        cv.circle(output, hip_point, radius, (255, 0, 0), 2)

        # draw circle for upper body
        radius = np.linalg.norm(shoulder_point - upper_body).astype(np.int32)
        cv.circle(output, shoulder_point, radius, (0, 255, 255), 2)

        # draw points for each keypoint
        for p in person_landmarks:
            cv.circle(output, p, 2, (0, 0, 255), 2)

        # put score
        cv.putText(output, 'Score: {:.4f}'.format(score), (0, output.shape[0] - 48), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

    cv.putText(output, 'Yellow: upper body circle', (0, output.shape[0] - 36), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255))
    cv.putText(output, 'Blue: full body circle', (0, output.shape[0] - 24), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
    cv.putText(output, 'Red: keypoint', (0, output.shape[0] - 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

    return output

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate MPPersonDet
    model = MPPersonDet(modelPath=args.model,
                      nmsThreshold=args.nms_threshold,
                      scoreThreshold=args.score_threshold,
                      topK=args.top_k,
                      backendId=backend_id,
                      targetId=target_id)

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)

        # Inference
        results = model.infer(image)
        if len(results) == 0:
            print('Person not detected')

        # Draw results on the input image
        image = visualize(image, results)

        # Save results if save is true
        if args.save:
            print('Resutls saved to result.jpg\n')
            cv.imwrite('result.jpg', image)

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, image)
            cv.waitKey(0)
    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Inference
            tm.start()
            results = model.infer(frame)
            tm.stop()

            # Draw results on the input image
            frame = visualize(frame, results, fps=tm.getFPS())

            # Visualize results in a new Window
            cv.imshow('MPPersonDet Demo', frame)

            tm.reset()

