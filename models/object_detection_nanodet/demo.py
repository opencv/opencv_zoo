import numpy as np
import cv2
import argparse

from nanodet import NanoDet

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

def letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()

    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv2.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv2.BORDER_CONSTANT, value=0)  # add border
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv2.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    letterbox_scale = [top, left, newh, neww]
    return img, letterbox_scale

def unletterbox(bbox, original_image_shape, letterbox_scale):
    ret = bbox.copy()

    h, w = original_image_shape
    top, left, newh, neww = letterbox_scale

    if h == w:
        ratio = h / newh
        ret = ret * ratio
        return ret

    ratioh, ratiow = h / newh, w / neww
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)
    ret[2] = min((ret[2] - left) * ratiow, w)
    ret[3] = min((ret[3] - top) * ratioh, h)

    return ret.astype(np.int32)

def vis(preds, res_img, letterbox_scale, fps=None):
    ret = res_img.copy()

    # draw FPS
    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv2.putText(ret, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # draw bboxes and labels
    for pred in preds:
        bbox = pred[:4]
        conf = pred[-2]
        classid = pred[-1].astype(np.int32)

        # bbox
        xmin, ymin, xmax, ymax = unletterbox(bbox, ret.shape[:2], letterbox_scale)
        cv2.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

        # label
        label = "{:s}: {:.2f}".format(classes[classid], conf)
        cv2.putText(ret, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    return ret

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--input', '-i', type=str, help='Path to the input image. Omit for using default camera.')
    parser.add_argument('--model', '-m', type=str, default='object_detection_nanodet_2022nov.onnx', help="Path to the model")
    parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
    parser.add_argument('--target', '-t', type=int, default=targets[0], help=help_msg_targets.format(*targets))
    parser.add_argument('--confidence', default=0.35, type=float, help='Class confidence')
    parser.add_argument('--nms', default=0.6, type=float, help='Enter nms IOU threshold')
    parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
    parser.add_argument('--vis', '-v', type=str2bool, default=True, help='Set true to open a window for result visualization. This flag is invalid when using camera.')
    args = parser.parse_args()

    model = NanoDet(modelPath= args.model,
                    prob_threshold=args.confidence,
                    iou_threshold=args.nms,
                    backend_id=args.backend,
                    target_id=args.target)

    tm = cv2.TickMeter()
    tm.reset()
    if args.input is not None:
        image = cv2.imread(args.input)
        input_blob = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Letterbox transformation
        input_blob, letterbox_scale = letterbox(input_blob)

        # Inference
        tm.start()
        preds = model.infer(input_blob)
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
            preds = model.infer(input_blob)
            tm.stop()

            img = vis(preds, frame, letterbox_scale, fps=tm.getFPS())

            cv2.imshow("NanoDet Demo", img)

            tm.reset()
