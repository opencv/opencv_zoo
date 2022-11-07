import cv2
import numpy as np
import argparse
import time
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

classes = (    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            )

def vis(preds, res_img):
    if preds is not None:
        image_shape = (416, 416)
        top, left, newh, neww = 0, 0, image_shape[0], image_shape[1]
        hw_scale = res_img.shape[0] / res_img.shape[1]
        if hw_scale > 1:
            newh, neww = image_shape[0], int(image_shape[1] / hw_scale)
            left = int((image_shape[1] - neww) * 0.5)
        else:
            newh, neww = int(image_shape[0] * hw_scale), image_shape[1]
            top = int((image_shape[0] - newh) * 0.5)

        ratioh,ratiow = res_img.shape[0]/newh,res_img.shape[1]/neww

        det_bboxes = preds[0]
        det_conf = preds[1]
        det_classid = preds[2]

        for i in range(det_bboxes.shape[0]):
            xmin, ymin, xmax, ymax = max(int((det_bboxes[i,0] - left) * ratiow), 0), max(int((det_bboxes[i,1] - top) * ratioh), 0), min(
            int((det_bboxes[i,2] - left) * ratiow), res_img.shape[1]), min(int((det_bboxes[i,3] - top) * ratioh), res_img.shape[0])
            cv2.rectangle(res_img, (xmin, ymin), (xmax, ymax), (0, 0, 0), thickness=2)
            #label = '%.2f' % det_conf[i]
            label=''
            label = '%s%s' % (classes[det_classid[i]], label)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
            cv2.putText(res_img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    else:
        print('No detections')

    return res_img

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

        a = time.time()
        preds = model_net.infer(image)
        b = time.time()
        print('Inference_Time:'+str(b-a)+' secs')

        srcimg = vis(preds, image)

        srcimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        cv2.namedWindow(args.image_path, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(args.image_path, srcimg)
        cv2.waitKey(0)

        if args.save:
            print('Resutls saved to result.jpg\n')
            cv2.imwrite('result.jpg', srcimg)

    else:
        print("Press 1 to stop video capture")
        cap = cv2.VideoCapture(0)
        tm = cv2.TickMeter()
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        total_frames = 0

        if(args.save):
            result = cv2.VideoWriter('Webcam_result.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, size)

        while cv2.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            frame = cv2.flip(frame, 1)
            #frame = cv2.resize(frame, [args.width, args.height])
            # Inference
            tm.start()
            preds = model_net.infer(frame)
            tm.stop()

            srcimg = vis(preds, frame)

            total_frames += 1
            fps=tm.getFPS()

            if fps > 0:
                fps_label = "FPS: %.2f" % fps
                cv2.putText(srcimg, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("output", srcimg)

            if cv2.waitKey(1) < 0:
                print("Stream terminated")
                break

            if(args.save):
                result.write(frame)

        print("Total frames: " + str(total_frames))
