import sys
import argparse

import numpy as np
import cv2 as cv

from mp_handpose import MPHandPose

sys.path.append('../palm_detection_mediapipe')
from mp_palmdet import MPPalmDet

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

backends = [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_CUDA]
targets = [cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16]
help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"
try:
    backends += [cv.dnn.DNN_BACKEND_TIMVX]
    targets += [cv.dnn.DNN_TARGET_NPU]
    help_msg_backends += "; {:d}: TIMVX"
    help_msg_targets += "; {:d}: NPU"
except:
    print('This version of OpenCV does not support TIM-VX and NPU. Visit https://gist.github.com/fengyuentau/5a7a5ba36328f2b763aea026c43fa45f for more information.')

parser = argparse.ArgumentParser(description='Hand Pose Estimation from MediaPipe')
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='./handpose_estimation_mediapipe_2022may.onnx', help='Path to the model.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--conf_threshold', type=float, default=0.8, help='Filter out hands of confidence < conf_threshold.')
parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()


def visualize(image, conf, hand_box, hand_landmarks, fps=None):
    output = image.copy()

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    landmarks = hand_landmarks.reshape(21, 2).astype(np.int32)

    # Draw line between each key points
    cv.line(output, landmarks[0], landmarks[1], (255, 255, 255), 2)
    cv.line(output, landmarks[1], landmarks[2], (255, 255, 255), 2)
    cv.line(output, landmarks[2], landmarks[3], (255, 255, 255), 2)
    cv.line(output, landmarks[3], landmarks[4], (255, 255, 255), 2)

    cv.line(output, landmarks[0], landmarks[5], (255, 255, 255), 2)
    cv.line(output, landmarks[5], landmarks[6], (255, 255, 255), 2)
    cv.line(output, landmarks[6], landmarks[7], (255, 255, 255), 2)
    cv.line(output, landmarks[7], landmarks[8], (255, 255, 255), 2)

    cv.line(output, landmarks[0], landmarks[9], (255, 255, 255), 2)
    cv.line(output, landmarks[9], landmarks[10], (255, 255, 255), 2)
    cv.line(output, landmarks[10], landmarks[11], (255, 255, 255), 2)
    cv.line(output, landmarks[11], landmarks[12], (255, 255, 255), 2)

    cv.line(output, landmarks[0], landmarks[13], (255, 255, 255), 2)
    cv.line(output, landmarks[13], landmarks[14], (255, 255, 255), 2)
    cv.line(output, landmarks[14], landmarks[15], (255, 255, 255), 2)
    cv.line(output, landmarks[15], landmarks[16], (255, 255, 255), 2)

    cv.line(output, landmarks[0], landmarks[17], (255, 255, 255), 2)
    cv.line(output, landmarks[17], landmarks[18], (255, 255, 255), 2)
    cv.line(output, landmarks[18], landmarks[19], (255, 255, 255), 2)
    cv.line(output, landmarks[19], landmarks[20], (255, 255, 255), 2)

    for p in landmarks:
        cv.circle(output, p, 2, (0, 0, 255), 2)

    return output


if __name__ == '__main__':
    # palm detector
    palm_detector = MPPalmDet(modelPath='../palm_detection_mediapipe/palm_detection_mediapipe_2022may.onnx',
                              nmsThreshold=0.3,
                              scoreThreshold=0.8,
                              backendId=args.backend,
                              targetId=args.target)
    # handpose detector
    handpose_detector = MPHandPose(modelPath=args.model,
                                   confThreshold=args.conf_threshold,
                                   backendId=args.backend,
                                   targetId=args.target)

    # If input is an image
    if args.input is not None:
        image = cv.imread(args.input)

        # Palm detector inference
        score, palm_box, palm_landmarks = palm_detector.infer(image)
        if score is None or palm_box is None or palm_landmarks is None:
            print('Palm not detected!')
        else:
            palm_box = palm_box.reshape(2, 2)
            # Handpose detector inference
            conf, hand_box, hand_landmarks = handpose_detector.infer(image, palm_box, palm_landmarks)

            # Print results
            print('Conf: {}, bbox: {}, landmarks: {}'.format(conf, hand_box, hand_landmarks))

            # Draw results on the input image
            if hand_landmarks is not None:
                frame = visualize(image, conf, hand_box, hand_landmarks)
            else:
                print('The conf: {:.2f} is lower than threshold.'.format(conf))

            # Save results
            if args.save:
                cv.imwrite('result.jpg', image)
                print('Results saved to result.jpg\n')

            # Visualize results in a new window
            if args.vis:
                cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
                cv.imshow(args.input, image)
                cv.waitKey(0)
    else:  # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Palm detector inference
            tm.start()
            score, palm_box, palm_landmarks = palm_detector.infer(frame)
            tm.stop()

            if score is None or palm_box is None or palm_landmarks is None:
                print('No palm detected!')
            else:
                palm_box = palm_box.reshape(2, 2)
                conf, hand_box, hand_landmarks = handpose_detector.infer(frame, palm_box, palm_landmarks)
                if hand_landmarks is not None:
                    frame = visualize(frame, conf, hand_box, hand_landmarks, fps=tm.getFPS())
                else:
                    print('The conf: {:.2f} is lower than threshold.'.format(conf))

            cv.imshow('MediaPipe Handpose Detection Demo', frame)
            tm.reset()
