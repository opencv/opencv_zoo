import cv2
import argparse
import numpy as np

config = {'person_conf_thres': 0.7, 'person_iou_thres': 0.45, 'kp_conf_thres': 0.5,
          'kp_iou_thres': 0.45, 'conf_thres_kp_person': 0.2, 'overwrite_tol': 25,
          'kp_face': [0, 1, 2, 3, 4], 'use_kp_dets': True,
          'segments': {1: [5, 6], 2: [5, 11], 3: [11, 12], 4: [12, 6], 5: [5, 7], 6: [7, 9], 7: [6, 8], 8: [8, 10],
                       9: [11, 13], 10: [13, 15], 11: [12, 14], 12: [14, 16]},
          'crowd_segments':{1: [0, 13], 2: [1, 13], 3: [0, 2], 4: [2, 4], 5: [1, 3], 6: [3, 5], 7: [0, 6], 8: [6, 7], 9: [7, 1], 10: [6, 8], 11: [8, 10], 12: [7, 9], 13: [9, 11], 14: [12, 13]},
          'crowd_kp_face':[]}

class MCN():
    def __init__(self, modelpath):
        with open('class.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
            self.lines = config['segments']
            self.kp_face = config['kp_face']

        self.num_classes = len(self.classes)
        self.inpHeight, self.inpWidth = 1280, 1280
        anchors = [[19, 27, 44, 40, 38, 94], [96, 68, 86, 152, 180, 137], [140, 301, 303, 264, 238, 542],
                   [436, 615, 739, 380, 925, 792]]
        self.stride = np.array([8., 16., 32., 64.])
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [np.zeros(1)] * self.nl
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.net = cv2.dnn.readNet(modelpath)
        self._inputNames = ''
        self.last_ind = 5 + self.num_classes

    def resize_image(self, srcimg, keep_ratio=True, dynamic=False):
        top, left, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                if not dynamic:
                    left = int((self.inpWidth - neww) * 0.5)
                    img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                             value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                if not dynamic:
                    top = int((self.inpHeight - newh) * 0.5)
                    img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                             value=(114, 114, 114))
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def postprocess(self, frame, outs, padsize=None):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        newh, neww, padh, padw = padsize
        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.

        person_confidences, kp_confidences = [], []
        person_boxes, kp_boxes = [], []
        person_classIds, kp_classIds = [], []
        person_rowinds = []
        for i in range(outs.shape[0]):
            detection = outs[i, :]
            scores = detection[5:self.last_ind]
            classId = np.argmax(scores)
            confidence = scores[classId] * detection[4]
            if classId == 0:
                if detection[4] > config['person_conf_thres'] and confidence > config['person_conf_thres']:
                    center_x = int((detection[0] - padw) * ratiow)
                    center_y = int((detection[1] - padh) * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width * 0.5)
                    top = int(center_y - height * 0.5)

                    person_confidences.append(float(confidence))
                    person_boxes.append([left, top, width, height])
                    person_classIds.append(classId)
                    person_rowinds.append(i)
            else:
                if detection[4] > config['kp_conf_thres'] and confidence > config['kp_conf_thres']:
                    center_x = int((detection[0] - padw) * ratiow)
                    center_y = int((detection[1] - padh) * ratioh)
                    width = int(detection[2] * ratiow)
                    height = int(detection[3] * ratioh)
                    left = int(center_x - width * 0.5)
                    top = int(center_y - height * 0.5)

                    kp_confidences.append(float(confidence))
                    kp_boxes.append([left, top, width, height])
                    kp_classIds.append(classId)

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        # print(person_boxes)
        if len(person_boxes) == 0:
            return frame
        person_indices = cv2.dnn.NMSBoxes(person_boxes, person_confidences, config['person_conf_thres'],
                                          config['person_iou_thres']).flatten()
        kp_indices = cv2.dnn.NMSBoxes(kp_boxes, kp_confidences, config['kp_conf_thres'],
                                      config['kp_iou_thres']).flatten()

        poses = []
        for i in person_indices:
            if person_confidences[i] > config['conf_thres_kp_person']:
                pose = outs[person_rowinds[i], self.last_ind:].reshape((-1, 2))
                pose[:, 0] = (pose[:, 0] - padw) * ratiow
                pose[:, 1] = (pose[:, 1] - padh) * ratioh
                poses.append(pose)
        nd = len(poses)
        poses = np.array(poses)
        poses = np.concatenate((poses, np.zeros((nd, poses.shape[1], 1))), axis=-1)
        for j in kp_indices:
            box = kp_boxes[j]
            x = box[0] + 0.5 * box[2]
            y = box[1] + 0.5 * box[3]
            pt_id = kp_classIds[j] - 1
            pose_kps = poses[:, pt_id, :]
            dist = np.linalg.norm(pose_kps[:, :2] - np.array([[x, y]]), axis=-1)
            kp_match = np.argmin(dist)
            if kp_confidences[j] > pose_kps[kp_match, 2] and dist[kp_match] < config['overwrite_tol']:
                poses[kp_match, pt_id, :] = np.array([x, y, kp_confidences[j]])

        for i in person_indices:
            box = person_boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, person_classIds[i], person_confidences[i], left, top, left + width,
                                  top + height)

        for pose in poses:
            for seg in self.lines.values():
                pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                cv2.line(frame, pt1, pt2, (255, 0, 255), 1)
            for x, y, c in pose:
                if c > 0:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 1)

            #for x, y, c in pose[self.kp_face]:
                #cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 255), 1)
        # for i in kp_indices:
        #     box = kp_boxes[i]
        #     left = box[0]
        #     top = box[1]
        #     width = box[2]
        #     height = box[3]
        #     frame = self.drawPred(frame, kp_classIds[i], kp_confidences[i], left, top, left + width, top + height)
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=1)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        return frame

    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, swapRB=True)
        # blob = cv2.dnn.blobFromImage(self.preprocess(img))
        # Sets the input to the network
        self.net.setInput(blob, self._inputNames)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0].squeeze(axis=0)

        # inference output
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h, w):
                self.grid[i] = self._make_grid(w, h)

            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                self.grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)

            self.num_coords = outs.shape[1] - self.last_ind
            outs[row_ind:row_ind + length, self.last_ind:] = outs[row_ind:row_ind + length, self.last_ind:] * 4. - 2.
            outs[row_ind:row_ind + length, self.last_ind:] *= np.tile(np.repeat(self.anchor_grid[i], h * w, axis=0), (1, self.num_coords//2))
            outs[row_ind:row_ind + length, self.last_ind:] += np.tile(np.tile(self.grid[i], (self.na, 1)) * int(self.stride[i]), (1, self.num_coords//2))
            row_ind += length
        srcimg = self.postprocess(srcimg, outs, padsize=(newh, neww, padh, padw))
        return srcimg
