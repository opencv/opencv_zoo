import sys
import argparse
import math
import numpy as np
import cv2 as cv

from mp_handpose import MPHandPose

sys.path.append('../palm_detection_mediapipe')
from mp_palmdet import MPPalmDet

dict = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


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
    print(
        'This version of OpenCV does not support TIM-VX and NPU. Visit https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU for more information.')

parser = argparse.ArgumentParser(description='Hand Pose Estimation from MediaPipe')
parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='./handpose_estimation_mediapipe_2023feb.onnx',
                    help='Path to the model.')
parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Filter out hands of confidence < conf_threshold.')
parser.add_argument('--save', '-s', type=str, default=False,
                    help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', type=str2bool, default=True,
                    help='Set true to open a window for result visualization. This flag is invalid when using camera.')
args = parser.parse_args()


def visualize(image, hands, print_result=False):
    display_screen = image.copy()
    display_3d = np.zeros((400, 400, 3), np.uint8)
    cv.line(display_3d, (200, 0), (200, 400), (255, 255, 255), 2)
    cv.line(display_3d, (0, 200), (400, 200), (255, 255, 255), 2)
    cv.putText(display_3d, 'Main View', (0, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Top View', (200, 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Left View', (0, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv.putText(display_3d, 'Right View', (200, 212), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    is_draw = False  # ensure only one hand is drawn

    def draw_lines(image, landmarks, is_draw_point=True, thickness=2):
        cv.line(image, landmarks[0], landmarks[1], (255, 255, 255), thickness)
        cv.line(image, landmarks[1], landmarks[2], (255, 255, 255), thickness)
        cv.line(image, landmarks[2], landmarks[3], (255, 255, 255), thickness)
        cv.line(image, landmarks[3], landmarks[4], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[5], (255, 255, 255), thickness)
        cv.line(image, landmarks[5], landmarks[6], (255, 255, 255), thickness)
        cv.line(image, landmarks[6], landmarks[7], (255, 255, 255), thickness)
        cv.line(image, landmarks[7], landmarks[8], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[9], (255, 255, 255), thickness)
        cv.line(image, landmarks[9], landmarks[10], (255, 255, 255), thickness)
        cv.line(image, landmarks[10], landmarks[11], (255, 255, 255), thickness)
        cv.line(image, landmarks[11], landmarks[12], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[13], (255, 255, 255), thickness)
        cv.line(image, landmarks[13], landmarks[14], (255, 255, 255), thickness)
        cv.line(image, landmarks[14], landmarks[15], (255, 255, 255), thickness)
        cv.line(image, landmarks[15], landmarks[16], (255, 255, 255), thickness)

        cv.line(image, landmarks[0], landmarks[17], (255, 255, 255), thickness)
        cv.line(image, landmarks[17], landmarks[18], (255, 255, 255), thickness)
        cv.line(image, landmarks[18], landmarks[19], (255, 255, 255), thickness)
        cv.line(image, landmarks[19], landmarks[20], (255, 255, 255), thickness)

        if is_draw_point:
            for p in landmarks:
                cv.circle(image, p, thickness, (0, 0, 255), -1)

    for idx, handpose in enumerate(hands):
        conf = handpose[-1]
        bbox = handpose[0:4].astype(np.int32)
        handedness = handpose[-2]
        if handedness <= 0.5:
            handedness_text = 'Left'
        else:
            handedness_text = 'Right'
        landmarks_screen = handpose[4:67].reshape(21, 3).astype(np.int32)
        landmarks_word = handpose[67:130].reshape(21, 3)

        # Print results
        if print_result:
            print('-----------hand {}-----------'.format(idx + 1))
            print('conf: {:.2f}'.format(conf))
            print('handedness: {}'.format(handedness_text))
            print('hand box: {}'.format(bbox))
            print('hand landmarks: ')
            for l in landmarks_screen:
                print('\t{}'.format(l))
            print('hand world landmarks: ')
            for l in landmarks_word:
                print('\t{}'.format(l))

        # draw box
        cv.rectangle(display_screen, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # draw handedness
        cv.putText(display_screen, '{}'.format(handedness_text), (bbox[0], bbox[1] + 12), cv.FONT_HERSHEY_DUPLEX, 0.5,
                   (0, 0, 255))
        # Draw line between each key points
        landmarks_xy = landmarks_screen[:, 0:2]
        draw_lines(display_screen, landmarks_xy, is_draw_point=False)

        # z value is relative to WRIST
        for p in landmarks_screen:
            r = max(5 - p[2] // 5, 0)
            r = min(r, 14)
            cv.circle(display_screen, np.array([p[0], p[1]]), r, (0, 0, 255), -1)

        if is_draw is False:
            is_draw = True
            # Main view
            landmarks_xy = landmarks_word[:, [0, 1]]
            landmarks_xy = (landmarks_xy * 1000 + 100).astype(np.int32)
            draw_lines(display_3d, landmarks_xy, thickness=5)

            # Top view
            landmarks_xz = landmarks_word[:, [0, 2]]
            landmarks_xz[:, 1] = -landmarks_xz[:, 1]
            landmarks_xz = (landmarks_xz * 1000 + np.array([300, 100])).astype(np.int32)
            draw_lines(display_3d, landmarks_xz, thickness=5)

            # Left view
            landmarks_yz = landmarks_word[:, [2, 1]]
            landmarks_yz[:, 0] = -landmarks_yz[:, 0]
            landmarks_yz = (landmarks_yz * 1000 + np.array([100, 300])).astype(np.int32)
            draw_lines(display_3d, landmarks_yz, thickness=5)

            # Right view
            landmarks_zy = landmarks_word[:, [2, 1]]
            landmarks_zy = (landmarks_zy * 1000 + np.array([300, 300])).astype(np.int32)
            draw_lines(display_3d, landmarks_zy, thickness=5)

    return display_screen, display_3d


def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos(
            (v1_x * v2_x + v1_y * v2_y) / (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))))
    except:
        angle_ = 65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_


def hand_angle(hand_):
    angle_list = []
    # ---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    return angle_list


def fingerStatus(lmList):
    fingerList = []
    originx, originy = lmList[0]
    keypoint_list = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    for point in keypoint_list:
        x1, y1 = lmList[point[0]]
        x2, y2 = lmList[point[1]]
        if math.hypot(x2 - originx, y2 - originy) > math.hypot(x1 - originx, y1 - originy):
            fingerList.append(True)
        else:
            fingerList.append(False)

    return fingerList


def h_gesture(hand):
    thr_angle = 65.
    thr_angle_thumb = 30.
    thr_angle_s = 49.
    gesture_str = None

    angle_list = hand_angle(hand)

    if 65535. not in angle_list and len(hand) != 0:
        thumbOpen, firstOpen, secondOpen, thirdOpen, fourthOpen = fingerStatus(hand)
        # Number
        if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                not firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "zero"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "one"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                not thumbOpen and firstOpen and secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "two"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle) and \
                not thumbOpen and firstOpen and secondOpen and thirdOpen and not fourthOpen:
            gesture_str = "three"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle) and \
                firstOpen and secondOpen and thirdOpen and fourthOpen:
            gesture_str = "four"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and thirdOpen and fourthOpen:
            gesture_str = "five"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s) and \
                thumbOpen and not firstOpen and not secondOpen and not thirdOpen and fourthOpen:
            gesture_str = "six"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "seven"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] < thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "eight"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] < thr_angle) and (
                angle_list[3] < thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and thirdOpen and not fourthOpen:
            gesture_str = "nine"

    return gesture_str


if __name__ == '__main__':
    # palm detector
    palm_detector = MPPalmDet(modelPath='../palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx',
                              nmsThreshold=0.3,
                              scoreThreshold=0.6,
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
        palms = palm_detector.infer(image)
        hands = np.empty(shape=(0, 132))

        # Estimate the pose of each hand
        for palm in palms:
            # Handpose detector inference
            handpose = handpose_detector.infer(image, palm)
            if handpose is not None:
                hands = np.vstack((hands, handpose))
        # Draw results on the input image
        image, view_3d = visualize(image, hands, True)

        if len(palms) == 0:
            print('No palm detected!')
        else:
            print('Palm detected!')

        # Save results
        if args.save:
            cv.imwrite('result.jpg', image)
            print('Results saved to result.jpg\n')

        # Visualize results in a new window
        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, image)
            cv.imshow('3D HandPose Demo', view_3d)
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
            palms = palm_detector.infer(frame)
            hands = np.empty(shape=(0, 132))

            tm.start()
            # Estimate the pose of each hand
            for palm in palms:
                # Handpose detector inference
                handpose = handpose_detector.infer(frame, palm)
                if handpose is not None:
                    hands = np.vstack((hands, handpose))

                    for idx, handpose in enumerate(hands):
                        conf = handpose[-1]
                        bbox = handpose[0:4].astype(np.int32)
                        handedness = handpose[-2]

                        landmarks_screen = handpose[4:67].reshape(21, 3).astype(np.int32)
                        landmarks_word = handpose[67:130].reshape(21, 3)

                        landmarks = landmarks_word[:, [0, 1]]
                        landmarks = (landmarks * 1000 + 100).astype(np.int32)

                        hand_local = []
                        for i in range(21):
                            x = landmarks[i][0] * frame.shape[1]
                            y = landmarks[i][1] * frame.shape[0]
                            hand_local.append((x, y))

                        if hand_local:
                            gesture = h_gesture(hand_local)
                        if gesture is not None:
                            tmp = dict[gesture]
                            cv.putText(frame, gesture, (1000, 100), 0, 1.3, (0, 0, 255), 3)
            tm.stop()
            # Draw results on the input image
            frame, view_3d = visualize(frame, hands)

            if len(palms) == 0:
                print('No palm detected!')
            else:
                print('Palm detected!')
                cv.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            cv.imshow('MediaPipe Handpose Detection Demo', frame)
            cv.imshow('3D HandPose Demo', view_3d)
            tm.reset()
