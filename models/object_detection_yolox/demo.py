import numpy as np
import cv2
import argparse

from yolox import YoloX

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

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
    print('This version of OpenCV does not support TIM-VX and NPU. Visit https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU for more information.')

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def letterbox(srcimg, target_size=(640, 640)):
    padded_img = np.ones((target_size[0], target_size[1], 3)) * 114.0
    ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
    resized_img = cv2.resize(
        srcimg, (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)] = resized_img

    return padded_img, ratio

def unletterbox(bbox, letterbox_scale):
    return bbox / letterbox_scale

def vis(dets, srcimg, letterbox_scale, fps=None):
    res_img = srcimg.copy()

    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv2.putText(res_img, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for det in dets:
        box = unletterbox(det[:4], letterbox_scale).astype(np.int32)
        score = det[-2]
        cls_id = int(det[-1])

        x0, y0, x1, y1 = box

        text = '{}:{:.1f}%'.format(classes[cls_id], score * 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(res_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.rectangle(res_img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), (255, 255, 255), -1)
        cv2.putText(res_img, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 0, 0), thickness=1)

    return res_img

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
    parser.add_argument('--model', '-m', type=str, default='object_detection_yolox_2022nov.onnx', help="Path to the model")
    parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
    parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
    parser.add_argument('--confidence', default=0.5, type=float, help='Class confidence')
    parser.add_argument('--nms', default=0.5, type=float, help='Enter nms IOU threshold')
    parser.add_argument('--obj', default=0.5, type=float, help='Enter object threshold')
    parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
    parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
    args = parser.parse_args()

    model_net = YoloX(modelPath= args.model,
                      confThreshold=args.confidence,
                      nmsThreshold=args.nms,
                      objThreshold=args.obj,
                      backendId=args.backend,
                      targetId=args.target)

    tm = cv2.TickMeter()
    tm.reset()
    if args.input is not None:
        image = cv2.imread(args.input)
        input_blob = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_blob, letterbox_scale = letterbox(input_blob)

        # Inference
        tm.start()
        preds = model_net.infer(input_blob)
        tm.stop()
        print("Inference time: {:.2f} ms".format(tm.getTimeMilli()))

        img = vis(preds, image, letterbox_scale)

        if args.save:
            print('Resutls saved to result.jpg\n')
            cv2.imwrite('result.jpg', img)

        if args.vis:
            cv2.namedWindow(args.input, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(args.input, img)
            cv2.waitKey(0)

    else:
        print("Press any key to stop video capture")
        deviceId = 0
        cap = cv2.VideoCapture(deviceId)

        while cv2.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            input_blob = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_blob, letterbox_scale = letterbox(input_blob)

            # Inference
            tm.start()
            preds = model_net.infer(input_blob)
            tm.stop()

            img = vis(preds, frame, letterbox_scale, fps=tm.getFPS())

            cv2.imshow("YoloX Demo", img)

            tm.reset()
