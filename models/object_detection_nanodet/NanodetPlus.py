import cv2
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
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
    parser.add_argument('--model', type=str, default='object_detection_nanodet-plus-m-1.5x-416.onnx', help="Path to the model")
    parser.add_argument('--input_type', type=str, default='image', help="Input types: image or video")
    parser.add_argument('--image_path', type=str, default='test2.jpg', help="Image path")
    parser.add_argument('--confidence', default=0.35, type=float, help='Class confidence')
    parser.add_argument('--nms', default=0.6, type=float, help='Enter nms IOU threshold')
    parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This flag is invalid when using camera.')
    args = parser.parse_args()
    model_net = NanoDet(modelPath= args.model ,prob_threshold=args.confidence, iou_threshold=args.nms)

    if (args.input_type=="image"):
        image = cv2.imread(args.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        srcimg = image
        drawimg = srcimg.copy()
        a = time.time()
        #image = model_net.infer(image)
        left, top, ratioh, ratiow, det_bboxes, det_conf, det_classid = model_net.infer(image)
        b = time.time()
        print('Inference_Time:'+str(b-a)+' secs')
        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = max(int((det_bboxes[i,0] - left) * ratiow), 0), max(int((det_bboxes[i,1] - top) * ratioh), 0), min(
                int((det_bboxes[i,2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i,3] - top) * ratioh), srcimg.shape[0])
            frame = drawPred(drawimg, det_classid[i], det_conf[i], xmin, ymin, xmax, ymax)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.namedWindow(args.image_path, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(args.image_path, frame)
        cv2.waitKey(0)
        
        if args.save:
            print('Resutls saved to result.jpg\n')
            cv2.imwrite('result.jpg', frame)

    else:
        print("Press 1 to stop video capture")
        cap = cv2.VideoCapture(0)
        tm = cv2.TickMeter()
        total_frames = 0
        frame_list = []
        Video_save = False
        if(args.save):
            Video_save = True

        while cv2.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            frame = cv2.flip(frame, 1)
            srcimg = frame
            drawimg = srcimg.copy()
            #frame = cv2.resize(frame, [args.width, args.height])
            # Inference
            tm.start()
            left, top, ratioh, ratiow, det_bboxes, det_conf, det_classid = model_net.infer(frame)
            tm.stop()

            for i in range(det_bboxes.shape[0]):
                xmin, ymin, xmax, ymax = max(int((det_bboxes[i,0] - left) * ratiow), 0), max(int((det_bboxes[i,1] - top) * ratioh), 0), min(
                    int((det_bboxes[i,2] - left) * ratiow), srcimg.shape[1]), min(int((det_bboxes[i,3] - top) * ratioh), srcimg.shape[0])
                image = drawPred(drawimg, det_classid[i], det_conf[i], xmin, ymin, xmax, ymax)

            total_frames += 1
            fps=tm.getFPS()

            if fps > 0:
                fps_label = "FPS: %.2f" % fps
                cv2.putText(image, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("output", image)

            if cv2.waitKey(1) > -1:
                print("Stream terminated")
                break

            if(args.save):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame_list.append(image)

        if(Video_save):
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_list, fps=fps)
            clip.write_videofile('Webcam_result.mp4')

        print("Total frames: " + str(total_frames))
