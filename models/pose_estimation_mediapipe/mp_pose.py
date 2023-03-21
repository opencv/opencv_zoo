import numpy as np
import cv2 as cv

class MPPose:
    def __init__(self, modelPath, confThreshold=0.5, backendId=0, targetId=0):
        self.model_path = modelPath
        self.conf_threshold = confThreshold
        self.backend_id = backendId
        self.target_id = targetId

        self.input_size = np.array([256, 256])  # wh
        # RoI will be larger so the performance will be better, but preprocess will be slower. Default to 1.
        self.PERSON_BOX_PRE_ENLARGE_FACTOR = 1
        self.PERSON_BOX_ENLARGE_FACTOR = 1.25

        self.model = cv.dnn.readNet(self.model_path)
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)

    def _preprocess(self, image, person):
        '''
        Rotate input for inference.
        Parameters:
          image - input image of BGR channel order
          face_bbox - human face bounding box found in image of format [[x1, y1], [x2, y2]] (top-left and bottom-right points)
          person_landmarks - 4 landmarks (2 full body points, 2 upper body points) of shape [4, 2]
        Returns:
          rotated_person - rotated person image for inference
          rotate_person_bbox - person box of interest range
          angle - rotate angle for person
          rotation_matrix - matrix for rotation and de-rotation
          pad_bias - pad pixels of interest range
        '''
        # crop and pad image to interest range
        pad_bias = np.array([0, 0], dtype=np.int32)  # left, top
        person_keypoints = person[4: 12].reshape(-1, 2)
        mid_hip_point = person_keypoints[0]
        full_body_point = person_keypoints[1]
        # get RoI
        full_dist = np.linalg.norm(mid_hip_point - full_body_point)
        full_bbox = np.array([mid_hip_point - full_dist, mid_hip_point + full_dist], np.int32)
        # enlarge to make sure full body can be cover
        center_bbox = np.sum(full_bbox, axis=0) / 2
        wh_bbox = full_bbox[1] - full_bbox[0]
        new_half_size = wh_bbox * self.PERSON_BOX_PRE_ENLARGE_FACTOR / 2
        full_bbox = np.array([
            center_bbox - new_half_size,
            center_bbox + new_half_size], np.int32)

        person_bbox = full_bbox.copy()
        # refine person bbox
        person_bbox[:, 0] = np.clip(person_bbox[:, 0], 0, image.shape[1])
        person_bbox[:, 1] = np.clip(person_bbox[:, 1], 0, image.shape[0])
        # crop to the size of interest
        image = image[person_bbox[0][1]:person_bbox[1][1], person_bbox[0][0]:person_bbox[1][0], :]
        # pad to square
        left, top = person_bbox[0] - full_bbox[0]
        right, bottom = full_bbox[1] - person_bbox[1]
        image = cv.copyMakeBorder(image, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
        pad_bias += person_bbox[0] - [left, top]
        # compute rotation
        mid_hip_point -= pad_bias
        full_body_point -= pad_bias
        radians = np.pi / 2 - np.arctan2(-(full_body_point[1] - mid_hip_point[1]), full_body_point[0] - mid_hip_point[0])
        radians = radians - 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)
        #  get rotation matrix
        rotation_matrix = cv.getRotationMatrix2D(mid_hip_point, angle, 1.0)
        #  get rotated image
        rotated_image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        #  get landmark bounding box
        blob = cv.resize(rotated_image, dsize=self.input_size, interpolation=cv.INTER_AREA).astype(np.float32)
        rotated_person_bbox = np.array([[0, 0], [image.shape[1], image.shape[0]]], dtype=np.int32)
        blob = cv.cvtColor(blob, cv.COLOR_BGR2RGB)
        blob = blob / 255. # [0, 1]
        return blob[np.newaxis, :, :, :], rotated_person_bbox, angle, rotation_matrix, pad_bias

    def infer(self, image, person):
        h, w, _ = image.shape
        # Preprocess
        input_blob, rotated_person_bbox, angle, rotation_matrix, pad_bias = self._preprocess(image, person)

        # Forward
        self.model.setInput(input_blob)
        output_blob = self.model.forward(self.model.getUnconnectedOutLayersNames())

        # Postprocess
        results = self._postprocess(output_blob, rotated_person_bbox, angle, rotation_matrix, pad_bias, np.array([w, h]))
        return results # [bbox_coords, landmarks_coords, conf]

    def _postprocess(self, blob, rotated_person_bbox, angle, rotation_matrix, pad_bias, img_size):
        landmarks, conf, mask, heatmap, landmarks_word = blob

        conf = conf[0][0]
        if conf < self.conf_threshold:
            return None

        landmarks = landmarks[0].reshape(-1, 5)  # shape: (1, 195) -> (39, 5)
        landmarks_word = landmarks_word[0].reshape(-1, 3) # shape: (1, 117) -> (39, 3)

        # recover sigmoid score
        landmarks[:, 3:] = 1 / (1 + np.exp(-landmarks[:, 3:]))
        # TODO: refine landmarks with heatmap. reference: https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/src/blazepose_tfjs/detector.ts#L577-L582
        heatmap = heatmap[0]

        # transform coords back to the input coords
        wh_rotated_person_bbox = rotated_person_bbox[1] - rotated_person_bbox[0]
        scale_factor = wh_rotated_person_bbox / self.input_size
        landmarks[:, :2] = (landmarks[:, :2] - self.input_size / 2) * scale_factor
        landmarks[:, 2] = landmarks[:, 2] * max(scale_factor) # depth scaling
        coords_rotation_matrix = cv.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_landmarks = np.dot(landmarks[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks = np.c_[rotated_landmarks, landmarks[:, 2:]]
        rotated_landmarks_world = np.dot(landmarks_word[:, :2], coords_rotation_matrix[:, :2])
        rotated_landmarks_world = np.c_[rotated_landmarks_world, landmarks_word[:, 2]]
        # invert rotation
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
        center = np.append(np.sum(rotated_person_bbox, axis=0) / 2, 1)
        original_center = np.array([
            np.dot(center, inverse_rotation_matrix[0]),
            np.dot(center, inverse_rotation_matrix[1])])
        landmarks[:, :2] = rotated_landmarks[:, :2] + original_center + pad_bias

        # get bounding box from rotated_landmarks
        bbox = np.array([
            np.amin(landmarks[:, :2], axis=0),
            np.amax(landmarks[:, :2], axis=0)])  # [top-left, bottom-right]
        center_bbox = np.sum(bbox, axis=0) / 2
        wh_bbox = bbox[1] - bbox[0]
        new_half_size = wh_bbox * self.PERSON_BOX_ENLARGE_FACTOR / 2
        bbox = np.array([
            center_bbox - new_half_size,
            center_bbox + new_half_size])

        # invert rotation for mask
        mask = mask[0].reshape(256, 256) # shape: (1, 256, 256, 1) -> (256, 256)
        invert_rotation_matrix = cv.getRotationMatrix2D((mask.shape[1]/2, mask.shape[0]/2), -angle, 1.0)
        invert_rotation_mask = cv.warpAffine(mask, invert_rotation_matrix, (mask.shape[1], mask.shape[0]))
        # enlarge mask
        invert_rotation_mask = cv.resize(invert_rotation_mask, wh_rotated_person_bbox)
        # crop and pad mask
        min_w, min_h = -np.minimum(pad_bias, 0)
        left, top = np.maximum(pad_bias, 0)
        pad_over = img_size - [invert_rotation_mask.shape[1], invert_rotation_mask.shape[0]] - pad_bias
        max_w, max_h = np.minimum(pad_over, 0) + [invert_rotation_mask.shape[1], invert_rotation_mask.shape[0]]
        right, bottom = np.maximum(pad_over, 0)
        invert_rotation_mask = invert_rotation_mask[min_h:max_h, min_w:max_w]
        invert_rotation_mask = cv.copyMakeBorder(invert_rotation_mask, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0)
        # binarize mask
        invert_rotation_mask = np.where(invert_rotation_mask > 0, 255, 0).astype(np.uint8)

        # 2*2 person bbox: [[x1, y1], [x2, y2]]
        # 39*5 screen landmarks: 33 keypoints and 6 auxiliary points with [x, y, z, visibility, presence], z value is relative to HIP
        # Visibility is probability that a keypoint is located within the frame and not occluded by another bigger body part or another object
        # Presence is probability that a keypoint is located within the frame
        # 39*3 world landmarks: 33 keypoints and 6 auxiliary points with [x, y, z] 3D metric x, y, z coordinate
        # img_height*img_width mask: gray mask, where 255 indicates the full body of a person and 0 means background
        # 64*64*39 heatmap: currently only used for refining landmarks, requires sigmod processing before use
        # conf: confidence of prediction
        return [bbox, landmarks, rotated_landmarks_world, invert_rotation_mask, heatmap, conf]
