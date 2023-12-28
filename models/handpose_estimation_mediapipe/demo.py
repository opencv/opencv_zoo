import sys
import argparse

import numpy as np
import cv2 as cv

from mp_handpose import MPHandPose

sys.path.append('../palm_detection_mediapipe')
from mp_palmdet import MPPalmDet

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

parser = argparse.ArgumentParser(description='Hand Pose Estimation from MediaPipe')
parser.add_argument('--input', '-i', type=str,
                    help='Path to the input image. Omit for using default camera.')
parser.add_argument('--model', '-m', type=str, default='./handpose_estimation_mediapipe_2023feb.onnx',
                    help='Path to the model.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Filter out hands of confidence < conf_threshold.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Specify to save results. This flag is invalid when using camera.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Specify to open a window for result visualization. This flag is invalid when using camera.')
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

    # used for gesture classification
    gc = GestureClassification()

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

        gesture = gc.classify(landmarks_screen)

        # Print results
        if print_result:
            print('-----------hand {}-----------'.format(idx + 1))
            print('conf: {:.2f}'.format(conf))
            print('handedness: {}'.format(handedness_text))
            print('gesture: {}'.format(gesture))
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
        cv.putText(display_screen, '{}'.format(handedness_text), (bbox[0], bbox[1] + 12), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        # draw gesture
        cv.putText(display_screen, '{}'.format(gesture), (bbox[0], bbox[1] + 30), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
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

class GestureClassification:
    def _vector_2_angle(self, v1, v2):
        uv1 = v1 / np.linalg.norm(v1)
        uv2 = v2 / np.linalg.norm(v2)
        angle = np.degrees(np.arccos(np.dot(uv1, uv2)))
        return angle

    def _hand_angle(self, hand):
        angle_list = []
        # thumb
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[2][0], hand[0][1] - hand[2][1]]),
            np.array([hand[3][0] - hand[4][0], hand[3][1] - hand[4][1]])
        )
        angle_list.append(angle_)
        # index
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[6][0], hand[0][1] - hand[6][1]]),
            np.array([hand[7][0] - hand[8][0], hand[7][1] - hand[8][1]])
        )
        angle_list.append(angle_)
        # middle
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[10][0], hand[0][1] - hand[10][1]]),
            np.array([hand[11][0] - hand[12][0], hand[11][1] - hand[12][1]])
        )
        angle_list.append(angle_)
        # ring
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[14][0], hand[0][1] - hand[14][1]]),
            np.array([hand[15][0] - hand[16][0], hand[15][1] - hand[16][1]])
        )
        angle_list.append(angle_)
        # pink
        angle_ = self._vector_2_angle(
            np.array([hand[0][0] - hand[18][0], hand[0][1] - hand[18][1]]),
            np.array([hand[19][0] - hand[20][0], hand[19][1] - hand[20][1]])
        )
        angle_list.append(angle_)
        return angle_list

    def _finger_status(self, lmList):
        fingerList = []
        originx, originy = lmList[0]
        keypoint_list = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
        for point in keypoint_list:
            x1, y1 = lmList[point[0]]
            x2, y2 = lmList[point[1]]
            if np.hypot(x2 - originx, y2 - originy) > np.hypot(x1 - originx, y1 - originy):
                fingerList.append(True)
            else:
                fingerList.append(False)

        return fingerList

    def _classify(self, hand):
        thr_angle = 65.
        thr_angle_thumb = 30.
        thr_angle_s = 49.
        gesture_str = "Undefined"

        angle_list = self._hand_angle(hand)

        thumbOpen, firstOpen, secondOpen, thirdOpen, fourthOpen = self._finger_status(hand)
        # Number
        if (angle_list[0] > thr_angle_thumb) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                not firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Zero"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "One"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle) and \
                not thumbOpen and firstOpen and secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Two"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle) and \
                not thumbOpen and firstOpen and secondOpen and thirdOpen and not fourthOpen:
            gesture_str = "Three"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle) and \
                firstOpen and secondOpen and thirdOpen and fourthOpen:
            gesture_str = "Four"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and thirdOpen and fourthOpen:
            gesture_str = "Five"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s) and \
                thumbOpen and not firstOpen and not secondOpen and not thirdOpen and fourthOpen:
            gesture_str = "Six"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] > thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and not secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Seven"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] < thr_angle) and (
                angle_list[3] > thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and not thirdOpen and not fourthOpen:
            gesture_str = "Eight"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle) and (angle_list[2] < thr_angle) and (
                angle_list[3] < thr_angle) and (angle_list[4] > thr_angle_s) and \
                thumbOpen and firstOpen and secondOpen and thirdOpen and not fourthOpen:
            gesture_str = "Nine"

        return gesture_str

    def classify(self, landmarks):
        hand = landmarks[:21, :2]
        gesture = self._classify(hand)
        return gesture

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    # palm detector
    palm_detector = MPPalmDet(modelPath='../palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx',
                              nmsThreshold=0.3,
                              scoreThreshold=0.6,
                              backendId=backend_id,
                              targetId=target_id)
    # handpose detector
    handpose_detector = MPHandPose(modelPath=args.model,
                                   confThreshold=args.conf_threshold,
                                   backendId=backend_id,
                                   targetId=target_id)

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
