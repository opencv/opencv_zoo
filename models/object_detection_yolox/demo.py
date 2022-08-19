import cv2
import numpy as np
import argparse
import time
import moviepy.video.io.ImageSequenceClip
from YoloX import YoloX

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


def vis(dets, res_img):
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        for i in range(len(final_boxes)):
            box = final_boxes[i]
            cls_id = int(final_cls_inds[i])
            score = final_scores[i]
            if score < args.confidence:
                continue

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            text = '{}:{:.1f}%'.format(classes[cls_id], score * 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(res_img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.rectangle(res_img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), (255, 255, 255), -1)
            cv2.putText(res_img, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 255, 0), thickness=1)

    else:
        print("No detections")

    return res_img

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--model', type=str, default='yolox_s.onnx', help="Path to the model")
    parser.add_argument('--input_type', type=str, default='image', help="Input types: image or video")
    parser.add_argument('--image_path', type=str, default='test2.jpg', help="Image path")
    parser.add_argument('--video_path', type=str, default='sample.mp4', help="Video path")
    parser.add_argument('--confidence', default=0.5, type=float, help='Class confidence')
    parser.add_argument('--nms', default=0.5, type=float, help='Enter nms IOU threshold')
    parser.add_argument('--obj', default=0.5, type=float, help='Enter object threshold')
    parser.add_argument('--save', '-s', type=str, default=False, help='Set true to save results. This option is invalid when using camera.')
    args = parser.parse_args()

    model_net = YoloX(modelPath= args.model, confThreshold=args.confidence, nmsThreshold=args.nms, objThreshold=args.obj)
    
    if (args.input_type=="image"):
        srcimg = cv2.imread(args.image_path)

        image = srcimg
        a = time.time()
        preds = model_net.infer(srcimg)
        b = time.time()
        print('Inference_Time:'+str(b-a)+' secs')

        srcimg = vis(preds, image)

        cv2.namedWindow(args.image_path, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(args.image_path, srcimg)
        cv2.waitKey(0)

        if args.save:
            print('Resutls saved to result.jpg\n')
            cv2.imwrite('result.jpg', image)

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

            # Inference
            tm.start()
            preds = model_net.infer(srcimg)
            tm.stop()

            srcimg = vis(preds, srcimg)

            total_frames += 1
            fps=tm.getFPS()

            if fps > 0:
                fps_label = "FPS: %.2f" % fps
                cv2.putText(srcimg, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("output", srcimg)

            if cv2.waitKey(1) > -1:
                print("Stream terminated")
                break

            if(args.save):
                srcimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
                frame_list.append(srcimg)

        if(Video_save):
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_list, fps=fps)
            clip.write_videofile('Webcam_result.mp4')

        print("Total frames: " + str(total_frames))
