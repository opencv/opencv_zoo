import cv2
import argparse
import numpy as np
from multitask_centernet import MCN

config = {'person_conf_thres': 0.7, 'person_iou_thres': 0.45, 'kp_conf_thres': 0.5,
          'kp_iou_thres': 0.45, 'conf_thres_kp_person': 0.2, 'overwrite_tol': 25,
          'kp_face': [0, 1, 2, 3, 4], 'use_kp_dets': True,
          'segments': {1: [5, 6], 2: [5, 11], 3: [11, 12], 4: [12, 6], 5: [5, 7], 6: [7, 9], 7: [6, 8], 8: [8, 10],
                       9: [11, 13], 10: [13, 15], 11: [12, 14], 12: [14, 16]},
          'crowd_segments':{1: [0, 13], 2: [1, 13], 3: [0, 2], 4: [2, 4], 5: [1, 3], 6: [3, 5], 7: [0, 6], 8: [6, 7], 9: [7, 1], 10: [6, 8], 11: [8, 10], 12: [7, 9], 13: [9, 11], 14: [12, 13]},
          'crowd_kp_face':[]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/d2645891.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='MCN.onnx')
    args = parser.parse_args()

    mcn = MCN(args.modelpath)
    srcimg = cv2.imread(args.imgpath)
    srcimg = mcn.detect(srcimg)
    cv2.imwrite('result.png', srcimg)


    # winName = 'using MCN in OpenCV'
    # cv2.namedWindow(winName, 0)
    # cv2.imshow(winName, srcimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
