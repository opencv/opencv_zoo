import cv2
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from NanodetPlus import NanoDet

backends = [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_BACKEND_CUDA]
targets = [cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16]
help_msg_backends = "Choose one of the computation backends: {:d}: OpenCV implementation (default); {:d}: CUDA"
help_msg_targets = "Chose one of the target computation devices: {:d}: CPU (default); {:d}: CUDA; {:d}: CUDA fp16"

try:
    backends += [cv2.dnn.DNN_BACKEND_TIMVX]
    targets += [cv2.dnn.DNN_TARGET_NPU]
    help_msg_backends += "; {:d}: TIMVX"
    help_msg_targets += "; {:d}: NPU"
except:
    print('This version of OpenCV does not support TIM-VX and NPU. Visit https://gist.github.com/Sidd1609/5bb321c8733110ed613ec120c7c02e41 for more information.')

with open('coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def drawPred(frame, classId, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), thickness=2)
    #label = '%.2f' % conf
    label =''
    label = '%s%s' % (classes[classId], label)
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    return frame

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--input_type', type=str, default='image', help="Input types: image or video")
    parser.add_argument('--image_path', type=str, default='test2.jpg', help="image path")
    parser.add_argument('--video_path', type=str, default='sample.mp4', help="video path")
    parser.add_argument('--confidence', default=0.35, type=float, help='class confidence')
    parser.add_argument('--nms', default=0.6, type=float, help='enter nms IOU threshold')
    args = parser.parse_args()

    if (args.input_type=="image"):
        image = cv2.imread(args.image_path)
        image = cv2. cvtColor(image, cv2.COLOR_BGR2RGB)
        model_net = NanoDet(prob_threshold=args.confidence, iou_threshold=args.nms)

        srcimg = image
        drawimg = srcimg.copy()
        a = time.time()
        #image = model_net.infer(image)
        result, left, top, ratioh, ratiow, det_bboxes, det_conf, det_classid = model_net.infer(image)
        b = time.time()
        print('Inference_Time:'+str(b-a)+' secs')
        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = max(int((det_bboxes[i,0] - left) * ratiow), 0), max(int((det_bboxes[i,1] - top) * ratioh), 0), min(
                int((det_bboxes[i,2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i,3] - top) * ratioh), srcimg.shape[0])
            frame = drawPred(drawimg, det_classid[i], det_conf[i], xmin, ymin, xmax, ymax)

        plt.imshow(frame)
        plt.axis('off')
        plt.show()

    else:
        model_net = NanoDet(prob_threshold=args.confidence, iou_threshold=args.nms)
        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        capture = cv2.VideoCapture(args.video_path)
        start = time.time_ns()

        frame_count = 0
        total_frames = 0
        fps = -1

        model_net = NanoDet(prob_threshold=args.confidence, iou_threshold=args.nms)

        while True:
            hasframe, frame = capture.read()
            if not hasframe:
                print("End of stream")
                break

            #image = model_net.infer(frame)
            srcimg = frame
            drawimg = srcimg.copy()
            result, left, top, ratioh, ratiow, det_bboxes, det_conf, det_classid = model_net.infer(frame)
            for i in range(det_bboxes.shape[0]):
                xmin, ymin, xmax, ymax = max(int((det_bboxes[i,0] - left) * ratiow), 0), max(int((det_bboxes[i,1] - top) * ratioh), 0), min(
                    int((det_bboxes[i,2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i,3] - top) * ratioh), srcimg.shape[0])
                image = drawPred(drawimg, det_classid[i], det_conf[i], xmin, ymin, xmax, ymax)

            frame_count += 1
            total_frames += 1

            if frame_count >= 50:
                end = time.time_ns()
                fps = 1000000000 * frame_count / (end - start)
                frame_count = 0
                start = time.time_ns()

            if fps > 0:
                fps_label = "FPS: %.2f" % fps
                cv2.putText(image, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("output", image)
            if cv2.waitKey(1) > -1:
                print("finished by user")
                break

        print("Total frames: " + str(total_frames))
