import numpy as np
import cv2 as cv


class MPHandPose:
    def __init__(self, modelPath, confThreshold=0.8, backendId=0, targetId=0):
        self.model_path = modelPath
        self.conf_threshold = confThreshold
        self.backend_id = backendId
        self.target_id = targetId

        self.input_size = np.array([224, 224])  # wh
        self.PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2]
        self.PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0
        self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2
        self.PALM_BOX_SHIFT_VECTOR = [0, -0.4]
        self.PALM_BOX_ENLARGE_FACTOR = 3
        self.HAND_BOX_SHIFT_VECTOR = [0, -0.1]
        self.HAND_BOX_ENLARGE_FACTOR = 1.65

        self.model = cv.dnn.readNet(self.model_path)
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backendId):
        self.backend_id = backendId
        self.model.setPreferableBackend(self.backend_id)

    def setTarget(self, targetId):
        self.target_id = targetId
        self.model.setPreferableTarget(self.target_id)

    def _preprocess(self, image, palm):
        '''
        Rotate input for inference.
        Parameters:
          image - input image of BGR channel order
          palm_bbox - palm bounding box found in image of format [[x1, y1], [x2, y2]] (top-left and bottom-right points)
          palm_landmarks - 7 landmarks (5 finger base points, 2 palm base points) of shape [7, 2]
        Returns:
          rotated_hand - rotated hand image for inference
          rotation_matrix - matrix for rotation and de-rotation
        '''
        # Rotate input to have vertically oriented hand image
        #  compute rotation
        palm_bbox = palm[0:4].reshape(2, 2)
        palm_landmarks = palm[4:18].reshape(7, 2)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        p1 = palm_landmarks[self.PALM_LANDMARKS_INDEX_OF_PALM_BASE]
        p2 = palm_landmarks[self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE]
        radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        radians = radians - 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)
        #  get bbox center
        center_palm_bbox = np.sum(palm_bbox, axis=0) / 2
        #  get rotation matrix
        rotation_matrix = cv.getRotationMatrix2D(center_palm_bbox, angle, 1.0)
        #  get rotated image
        rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        #  get bounding boxes from rotated palm landmarks
        homogeneous_coord = np.c_[palm_landmarks, np.ones(palm_landmarks.shape[0])]
        rotated_palm_landmarks = np.array([
            np.dot(homogeneous_coord, rotation_matrix[0]),
            np.dot(homogeneous_coord, rotation_matrix[1])])
        #  get landmark bounding box
        rotated_palm_bbox = np.array([
            np.amin(rotated_palm_landmarks, axis=1),
            np.amax(rotated_palm_landmarks, axis=1)])  # [top-left, bottom-right]

        #  shift bounding box
        wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        shift_vector = self.PALM_BOX_SHIFT_VECTOR * wh_rotated_palm_bbox
        rotated_palm_bbox = rotated_palm_bbox + shift_vector
        #  squarify bounding boxx
        center_rotated_plam_bbox = np.sum(rotated_palm_bbox, axis=0) / 2
        wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        new_half_size = np.amax(wh_rotated_palm_bbox) / 2
        rotated_palm_bbox = np.array([
            center_rotated_plam_bbox - new_half_size,
            center_rotated_plam_bbox + new_half_size])

        #  enlarge bounding box
        center_rotated_plam_bbox = np.sum(rotated_palm_bbox, axis=0) / 2
        wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        new_half_size = wh_rotated_palm_bbox * self.PALM_BOX_ENLARGE_FACTOR / 2
        rotated_palm_bbox = np.array([
            center_rotated_plam_bbox - new_half_size,
            center_rotated_plam_bbox + new_half_size])

        # Crop and resize the rotated image by the bounding box
        [[x1, y1], [x2, y2]] = rotated_palm_bbox.astype(np.int32)
        diff = np.maximum([-x1, -y1, x2 - rotated_image.shape[1], y2 - rotated_image.shape[0]], 0)
        [x1, y1, x2, y2] = [x1, y1, x2, y2] + diff
        crop = rotated_image[y1:y2, x1:x2, :]
        crop = cv.copyMakeBorder(crop, diff[1], diff[3], diff[0], diff[2], cv.BORDER_CONSTANT, value=(0, 0, 0))
        blob = cv.resize(crop, dsize=self.input_size, interpolation=cv.INTER_AREA).astype(np.float32) / 255.0

        return blob[np.newaxis, :, :, :], rotated_palm_bbox, angle, rotation_matrix

    def infer(self, image, palm):
        # Preprocess
        input_blob, rotated_palm_bbox, angle, rotation_matrix = self._preprocess(image, palm)

        # Forward
        self.model.setInput(input_blob)
        output_blob = self.model.forward(self.model.getUnconnectedOutLayersNames())

        # Postprocess
        results = self._postprocess(output_blob, rotated_palm_bbox, angle, rotation_matrix)
        return results # [bbox_coords, landmarks_coords, conf]

    def _postprocess(self, blob, rotated_palm_bbox, angle, rotation_matrix):
        landmarks, conf, handedness, landmarks_word = blob

        conf = conf[0][0]
        if conf < self.conf_threshold:
            return None

        landmarks = landmarks[0].reshape(-1, 3)  # shape: (1, 63) -> (21, 3)
        landmarks_word = landmarks_word[0].reshape(-1, 3) # shape: (1, 63) -> (21, 3)

        # transform coords back to the input coords
        wh_rotated_palm_bbox = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        scale_factor = wh_rotated_palm_bbox / self.input_size
        landmarks[:, :2] = (landmarks[:, :2] - self.input_size / 2) * scale_factor
        landmarks[:, 2] = landmarks[:, 2] * max(scale_factor) # depth scaling
        coords_rotation_matrix = cv.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_landmarks = np.dot(landmarks[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks = np.c_[rotated_landmarks, landmarks[:, 2]]
        rotated_landmarks_world = np.dot(landmarks_word[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks_world = np.c_[rotated_landmarks_world, landmarks_word[:, 2]]
        #  invert rotation
        rotation_component = np.array([
            [rotation_matrix[0][0], rotation_matrix[1][0]],
            [rotation_matrix[0][1], rotation_matrix[1][1]]])
        translation_component = np.array([
            rotation_matrix[0][2], rotation_matrix[1][2]])
        inverted_translation = np.array([
            -np.dot(rotation_component[0], translation_component),
            -np.dot(rotation_component[1], translation_component)])
        inverse_rotation_matrix = np.c_[rotation_component, inverted_translation]
        #  get box center
        center = np.append(np.sum(rotated_palm_bbox, axis=0) / 2, 1)
        original_center = np.array([
            np.dot(center, inverse_rotation_matrix[0]),
            np.dot(center, inverse_rotation_matrix[1])])
        landmarks[:, :2] = rotated_landmarks[:, :2] + original_center

        # get bounding box from rotated_landmarks
        bbox = np.array([
            np.amin(landmarks[:, :2], axis=0),
            np.amax(landmarks[:, :2], axis=0)])  # [top-left, bottom-right]
        # shift bounding box
        wh_bbox = bbox[1] - bbox[0]
        shift_vector = self.HAND_BOX_SHIFT_VECTOR * wh_bbox
        bbox = bbox + shift_vector
        # enlarge bounding box
        center_bbox = np.sum(bbox, axis=0) / 2
        wh_bbox = bbox[1] - bbox[0]
        new_half_size = wh_bbox * self.HAND_BOX_ENLARGE_FACTOR / 2
        bbox = np.array([
            center_bbox - new_half_size,
            center_bbox + new_half_size])

        # [0: 4]: hand bounding box found in image of format [x1, y1, x2, y2] (top-left and bottom-right points)
        # [4: 67]: screen landmarks with format [x1, y1, z1, x2, y2 ... x21, y21, z21], z value is relative to WRIST
        # [67: 130]: world landmarks with format [x1, y1, z1, x2, y2 ... x21, y21, z21], 3D metric x, y, z coordinate
        # [130]: handedness, (left)[0, 1](right) hand
        # [131]: confidence
        return np.r_[bbox.reshape(-1), landmarks.reshape(-1), rotated_landmarks_world.reshape(-1), handedness[0][0], conf]
