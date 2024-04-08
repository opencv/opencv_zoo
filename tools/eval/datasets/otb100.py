import os
import json
import numpy as np
import cv2 as cv
from colorama import Style, Fore
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def overlap_ratio(rect1, rect2):
    """Calculate the Intersection over Union (IoU) overlap ratio between two sets of rectangles."""  
    tl = np.maximum(rect1[:, :2], rect2[:, :2])
    br = np.minimum(rect1[:, :2] + rect1[:, 2:] - 1.0, rect2[:, :2] + rect2[:, 2:] - 1.0)
    sz = np.maximum(br - tl + 1.0, 0)

    # Area
    intersection = np.prod(sz, axis=1)
    union = np.prod(rect1[:, 2:], axis=1) + np.prod(rect2[:, 2:], axis=1) - intersection
    iou = np.clip(intersection / union, 0, 1)
    return iou


def success_overlap(gt_bb, result_bb, n_frame):
    """Calculate the success rate based on the overlap ratio between ground truth and predicted bounding boxes."""
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    mask = np.sum(gt_bb[:, 2:] > 0, axis=1) == 2
    iou = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i, threshold in enumerate(thresholds_overlap):
        success[i] = np.sum(iou > threshold) / n_frame
    return success

def success_error(gt_center, result_center, thresholds, n_frame):
    """Calculate the success rate based on the error distance between ground truth and predicted bounding box centers."""
    success = np.zeros(len(thresholds))
    mask = np.sum(gt_center > 0, axis=1) == 2
    dist = np.linalg.norm(gt_center[mask] - result_center[mask], axis=1)
    for i, threshold in enumerate(thresholds):
        success[i] = np.sum(dist <= threshold) / n_frame
    return success

class OPEBenchmark:
    def __init__(self, dataset):
        self.dataset = dataset

    def convert_bb_to_center(self, bboxes):
        """Convert bounding box coordinates to centers."""
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        """Convert bounding box coordinates to normalized centers."""
        return self.convert_bb_to_center(bboxes) / (gt_wh + 1e-16)

    def evaluate(self, metric):
        """Evaluate the tracking performance based on the specified metric."""
        evaluation_ret = {}
        for video in self.dataset:
            gt_traj = np.array(video.gt_traj)
            tracker_traj = np.array(video.load_tracker())
            n_frame = len(gt_traj)
            if hasattr(video, 'absent'):
                gt_traj = gt_traj[video.absent == 1]
                tracker_traj = tracker_traj[video.absent == 1]
            if metric == 'success':
                evaluation_ret[video.name] = success_overlap(gt_traj, tracker_traj, n_frame)
            elif metric == 'precision':
                gt_center = self.convert_bb_to_center(gt_traj)
                tracker_center = self.convert_bb_to_center(tracker_traj)
                thresholds = np.arange(0, 51, 1)
                evaluation_ret[video.name] = success_error(gt_center, tracker_center, thresholds, n_frame)
            elif metric == 'norm_precision':
                gt_center_norm = self.convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
                tracker_center_norm = self.convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])
                thresholds = np.arange(0, 51, 1) / 100
                evaluation_ret[video.name] = success_error(gt_center_norm, tracker_center_norm, thresholds, n_frame)
        return {"tracker": evaluation_ret}

    def show_result(self, success, precision=None, norm_precision=None, show_video_level=False, height_threshold=0.6):
        tracker_auc = {tracker_name: np.mean(list(scores.values())) for tracker_name, scores in success.items()}
        tracker_auc = sorted(tracker_auc.items(), key=lambda x: x[1], reverse=True)[:20]
        tracker_names = [x[0] for x in tracker_auc]
        tracker_name_len = max(max(len(x) for x in success.keys()) + 2, 12)
        header = ("|{:^" + str(tracker_name_len) + "}|{:^9}|{:^11}|{:^16}|").format(
            "Tracker name", "AUC", "Precision", "Norm Precision")
        formatter = "|{:^" + str(tracker_name_len) + "}|{:^9.3f}|{:^11.3f}|{:^16.3f}|"

        print('-' * len(header))
        print(header)
        print('-' * len(header))

        for tracker_name in tracker_names:
            success_score = np.mean(list(success[tracker_name].values()))
            precision_score = np.mean(list(precision[tracker_name].values()), axis=0)[20] if precision else 0
            norm_precision_score = np.mean(list(norm_precision[tracker_name].values()), axis=0)[20] if norm_precision else 0
            print(formatter.format(tracker_name, success_score, precision_score, norm_precision_score))

        print('-' * len(header))

        if show_video_level and len(success) < 10 and precision and len(precision) < 10:
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")

            for tracker_name in success.keys():
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("success", "precision")

            print('-' * len(header1))
            print(header1)
            print('-' * len(header1))
            print(header2)
            print('-' * len(header1))

            for video, scores in success.items():
                row = "|{:^21}|".format(video)

                for tracker_name in tracker_names:
                    success_score = np.mean(success[tracker_name][video])
                    precision_score = np.mean(precision[tracker_name][video])
                    success_str = f'{success_score:.3f}' if success_score < height_threshold else f'{success_score:.3f}'
                    precision_str = f'{precision_score:.3f}' if precision_score < height_threshold else f'{precision_score:.3f}'
                    row += f"{success_str:^9}|{precision_str:^11}|"

                print(row)

            print('-' * len(header1))

class Video:
    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect, attr):
        self.name = name
        self.video_dir = video_dir
        self.init_rect = init_rect
        self.gt_traj = gt_rect
        self.attr = attr
        self.pred_trajs = {}
        self.img_names = [os.path.join(root, x) for x in img_names]
        self.imgs = None
        img = cv.imread(self.img_names[0])
        assert img is not None, self.img_names[0]
        self.width = img.shape[1]
        self.height = img.shape[0]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.imgs is None:
            return cv.imread(self.img_names[idx]), self.gt_traj[idx]
        else:
            return self.imgs[idx], self.gt_traj[idx]

    def __iter__(self):
        for i in range(len(self.img_names)):
            if self.imgs is not None:
                yield self.imgs[i], self.gt_traj[i]
            else:
                yield cv.imread(self.img_names[i]), self.gt_traj[i]

    def load_tracker(self):
        """Load tracker results from file."""
        traj_file = os.path.join("OTB_results", self.name+'.txt')
        if os.path.exists(traj_file):
            with open(traj_file, 'r') as f:
                pred_traj = [list(map(float, x.strip().split(','))) for x in f.readlines()]
                if len(pred_traj) != len(self.gt_traj):
                    print("tracker", len(pred_traj), len(self.gt_traj), self.name)
                else:
                    return pred_traj
        else:
            print(traj_file)

class OTBDATASET:
    def __init__(self, root):
        meta_data = {}
        for sequence_info in sequence_info_list:
            sequence_path = sequence_info['path']
            nz = sequence_info['nz']
            ext = sequence_info['ext']
            start_frame = sequence_info['startFrame']
            end_frame = sequence_info['endFrame']

            init_omit = 0
            if 'initOmit' in sequence_info:
                init_omit = sequence_info['initOmit']
            frames = [f'{root}/OTB100/{sequence_path}/{frame_num:0{nz}}.{ext}' for \
                      frame_num in range(start_frame+init_omit, end_frame+1)]
            anno_path = f'{root}/OTB100/{sequence_info["anno_path"]}'
            ground_truth_rect = load_text_numpy(str(anno_path), (',', None), np.float64)[init_omit:,:]

            meta_data[sequence_info['name']] = {}
            meta_data[sequence_info['name']]['video_dir'] = sequence_info['path']
            meta_data[sequence_info['name']]['init_rect'] = ground_truth_rect[0]
            meta_data[sequence_info['name']]['img_names'] = frames
            meta_data[sequence_info['name']]['gt_rect'] = ground_truth_rect
            meta_data[sequence_info['name']]['attr'] = [sequence_info["object_class"]]

        self.videos = {}
        for video in meta_data.keys():
            self.videos[video] = Video(video,
                                       root,
                                       meta_data[video]['video_dir'],
                                       meta_data[video]['init_rect'],
                                       meta_data[video]['img_names'],
                                       meta_data[video]['gt_rect'],
                                       meta_data[video]['attr'])
        self.attr = {'ALL': list(self.videos.keys())}
        all_attributes = [x.attr for x in self.videos.values()]
        all_attributes = set(sum(all_attributes, []))
        for attr_ in all_attributes:
            self.attr[attr_] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            sorted_keys = sorted(list(self.videos.keys()))
            return self.videos[sorted_keys[idx]]

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        sorted_keys = sorted(list(self.videos.keys()))
        for key in sorted_keys:
            yield self.videos[key]

def get_axis_aligned_bbox(region):
    """Converts a region to (cx, cy, w, h) representing an axis-aligned box."""
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x, y, w, h = region
        cx = x + w / 2
        cy = y + h / 2
    return cx, cy, w, h

def load_text_numpy(path, delimiter, dtype):
    for d in delimiter:
        try:
            ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
            return ground_truth_rect
        except:
            pass

    raise Exception('Could not read file {}'.format(path))

class OTB100:
    def __init__(self, root):
        # Go up one if directory is provided
        root = os.path.abspath(root)
        if root.endswith("OTB100"):
            root = os.path.dirname(root)

        self.dataset = OTBDATASET(root)

    @property
    def name(self):
        return self.__class__.__name__

    def eval(self, model):
        for video in tqdm(self.dataset, desc="Evaluating: ", total=100, ncols=100):
            pred_bboxes = []

            for idx, (img, gt_bbox) in enumerate(video):
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
                    model.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                else:
                    isLocated, bbox, score = model.infer(img)
                    pred_bbox = bbox

                pred_bboxes.append(pred_bbox)

            model_path = os.path.join('OTB_results')
            os.makedirs(model_path, exist_ok=True)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for bbox in pred_bboxes:
                    f.write(','.join(map(str, bbox)) + '\n')

    def print_result(self):
        benchmark = OPEBenchmark(self.dataset)
        num_cores = cpu_count()
        evaluation_results = {}
        metrics = ["success", "precision", "norm_precision"]
        for metric in metrics:
            with Pool(processes=min(num_cores, max(1, num_cores - 1))) as pool:
                for ret in pool.imap_unordered(benchmark.evaluate, [metric], 1):
                    evaluation_results[metric] = ret

        benchmark.show_result(**evaluation_results, show_video_level=False)

# Sourced from https://github.com/lpylpy0514/VitTracker
sequence_info_list = [
    {"name": "Basketball", "path": "Basketball/img", "startFrame": 1, "endFrame": 725, "nz": 4, "ext": "jpg", "anno_path": "Basketball/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Biker", "path": "Biker/img", "startFrame": 1, "endFrame": 142, "nz": 4, "ext": "jpg", "anno_path": "Biker/groundtruth_rect.txt",
        "object_class": "person head"},
    {"name": "Bird1", "path": "Bird1/img", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg", "anno_path": "Bird1/groundtruth_rect.txt",
        "object_class": "bird"},
    {"name": "Bird2", "path": "Bird2/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg", "anno_path": "Bird2/groundtruth_rect.txt",
        "object_class": "bird"},
    {"name": "BlurBody", "path": "BlurBody/img", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg", "anno_path": "BlurBody/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "BlurCar1", "path": "BlurCar1/img", "startFrame": 247, "endFrame": 988, "nz": 4, "ext": "jpg", "anno_path": "BlurCar1/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "BlurCar2", "path": "BlurCar2/img", "startFrame": 1, "endFrame": 585, "nz": 4, "ext": "jpg", "anno_path": "BlurCar2/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "BlurCar3", "path": "BlurCar3/img", "startFrame": 3, "endFrame": 359, "nz": 4, "ext": "jpg", "anno_path": "BlurCar3/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "BlurCar4", "path": "BlurCar4/img", "startFrame": 18, "endFrame": 397, "nz": 4, "ext": "jpg", "anno_path": "BlurCar4/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "BlurFace", "path": "BlurFace/img", "startFrame": 1, "endFrame": 493, "nz": 4, "ext": "jpg", "anno_path": "BlurFace/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "BlurOwl", "path": "BlurOwl/img", "startFrame": 1, "endFrame": 631, "nz": 4, "ext": "jpg", "anno_path": "BlurOwl/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Board", "path": "Board/img", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg", "anno_path": "Board/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Bolt", "path": "Bolt/img", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg", "anno_path": "Bolt/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Bolt2", "path": "Bolt2/img", "startFrame": 1, "endFrame": 293, "nz": 4, "ext": "jpg", "anno_path": "Bolt2/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Box", "path": "Box/img", "startFrame": 1, "endFrame": 1161, "nz": 4, "ext": "jpg", "anno_path": "Box/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Boy", "path": "Boy/img", "startFrame": 1, "endFrame": 602, "nz": 4, "ext": "jpg", "anno_path": "Boy/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Car1", "path": "Car1/img", "startFrame": 1, "endFrame": 1020, "nz": 4, "ext": "jpg", "anno_path": "Car1/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "Car2", "path": "Car2/img", "startFrame": 1, "endFrame": 913, "nz": 4, "ext": "jpg", "anno_path": "Car2/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "Car24", "path": "Car24/img", "startFrame": 1, "endFrame": 3059, "nz": 4, "ext": "jpg", "anno_path": "Car24/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "Car4", "path": "Car4/img", "startFrame": 1, "endFrame": 659, "nz": 4, "ext": "jpg", "anno_path": "Car4/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "CarDark", "path": "CarDark/img", "startFrame": 1, "endFrame": 393, "nz": 4, "ext": "jpg", "anno_path": "CarDark/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "CarScale", "path": "CarScale/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg", "anno_path": "CarScale/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "ClifBar", "path": "ClifBar/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg", "anno_path": "ClifBar/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Coke", "path": "Coke/img", "startFrame": 1, "endFrame": 291, "nz": 4, "ext": "jpg", "anno_path": "Coke/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Couple", "path": "Couple/img", "startFrame": 1, "endFrame": 140, "nz": 4, "ext": "jpg", "anno_path": "Couple/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Coupon", "path": "Coupon/img", "startFrame": 1, "endFrame": 327, "nz": 4, "ext": "jpg", "anno_path": "Coupon/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Crossing", "path": "Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4, "ext": "jpg", "anno_path": "Crossing/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Crowds", "path": "Crowds/img", "startFrame": 1, "endFrame": 347, "nz": 4, "ext": "jpg", "anno_path": "Crowds/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Dancer", "path": "Dancer/img", "startFrame": 1, "endFrame": 225, "nz": 4, "ext": "jpg", "anno_path": "Dancer/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Dancer2", "path": "Dancer2/img", "startFrame": 1, "endFrame": 150, "nz": 4, "ext": "jpg", "anno_path": "Dancer2/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "David", "path": "David/img", "startFrame": 300, "endFrame": 770, "nz": 4, "ext": "jpg", "anno_path": "David/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "David2", "path": "David2/img", "startFrame": 1, "endFrame": 537, "nz": 4, "ext": "jpg", "anno_path": "David2/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "David3", "path": "David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg", "anno_path": "David3/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Deer", "path": "Deer/img", "startFrame": 1, "endFrame": 71, "nz": 4, "ext": "jpg", "anno_path": "Deer/groundtruth_rect.txt",
        "object_class": "mammal"},
    {"name": "Diving", "path": "Diving/img", "startFrame": 1, "endFrame": 215, "nz": 4, "ext": "jpg", "anno_path": "Diving/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Dog", "path": "Dog/img", "startFrame": 1, "endFrame": 127, "nz": 4, "ext": "jpg", "anno_path": "Dog/groundtruth_rect.txt",
        "object_class": "dog"},
    {"name": "Dog1", "path": "Dog1/img", "startFrame": 1, "endFrame": 1350, "nz": 4, "ext": "jpg", "anno_path": "Dog1/groundtruth_rect.txt",
        "object_class": "dog"},
    {"name": "Doll", "path": "Doll/img", "startFrame": 1, "endFrame": 3872, "nz": 4, "ext": "jpg", "anno_path": "Doll/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "DragonBaby", "path": "DragonBaby/img", "startFrame": 1, "endFrame": 113, "nz": 4, "ext": "jpg", "anno_path": "DragonBaby/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Dudek", "path": "Dudek/img", "startFrame": 1, "endFrame": 1145, "nz": 4, "ext": "jpg", "anno_path": "Dudek/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "FaceOcc1", "path": "FaceOcc1/img", "startFrame": 1, "endFrame": 892, "nz": 4, "ext": "jpg", "anno_path": "FaceOcc1/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "FaceOcc2", "path": "FaceOcc2/img", "startFrame": 1, "endFrame": 812, "nz": 4, "ext": "jpg", "anno_path": "FaceOcc2/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Fish", "path": "Fish/img", "startFrame": 1, "endFrame": 476, "nz": 4, "ext": "jpg", "anno_path": "Fish/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "FleetFace", "path": "FleetFace/img", "startFrame": 1, "endFrame": 707, "nz": 4, "ext": "jpg", "anno_path": "FleetFace/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Football", "path": "Football/img", "startFrame": 1, "endFrame": 362, "nz": 4, "ext": "jpg", "anno_path": "Football/groundtruth_rect.txt",
        "object_class": "person head"},
    {"name": "Football1", "path": "Football1/img", "startFrame": 1, "endFrame": 74, "nz": 4, "ext": "jpg", "anno_path": "Football1/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Freeman1", "path": "Freeman1/img", "startFrame": 1, "endFrame": 326, "nz": 4, "ext": "jpg", "anno_path": "Freeman1/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Freeman3", "path": "Freeman3/img", "startFrame": 1, "endFrame": 460, "nz": 4, "ext": "jpg", "anno_path": "Freeman3/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Freeman4", "path": "Freeman4/img", "startFrame": 1, "endFrame": 283, "nz": 4, "ext": "jpg", "anno_path": "Freeman4/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Girl", "path": "Girl/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg", "anno_path": "Girl/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Girl2", "path": "Girl2/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg", "anno_path": "Girl2/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Gym", "path": "Gym/img", "startFrame": 1, "endFrame": 767, "nz": 4, "ext": "jpg", "anno_path": "Gym/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Human2", "path": "Human2/img", "startFrame": 1, "endFrame": 1128, "nz": 4, "ext": "jpg", "anno_path": "Human2/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Human3", "path": "Human3/img", "startFrame": 1, "endFrame": 1698, "nz": 4, "ext": "jpg", "anno_path": "Human3/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Human4", "path": "Human4/img", "startFrame": 1, "endFrame": 667, "nz": 4, "ext": "jpg", "anno_path": "Human4/groundtruth_rect.2.txt",
        "object_class": "person"},
    {"name": "Human5", "path": "Human5/img", "startFrame": 1, "endFrame": 713, "nz": 4, "ext": "jpg", "anno_path": "Human5/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Human6", "path": "Human6/img", "startFrame": 1, "endFrame": 792, "nz": 4, "ext": "jpg", "anno_path": "Human6/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Human7", "path": "Human7/img", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg", "anno_path": "Human7/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Human8", "path": "Human8/img", "startFrame": 1, "endFrame": 128, "nz": 4, "ext": "jpg", "anno_path": "Human8/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Human9", "path": "Human9/img", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg", "anno_path": "Human9/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Ironman", "path": "Ironman/img", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg", "anno_path": "Ironman/groundtruth_rect.txt",
        "object_class": "person head"},
    {"name": "Jogging_1", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg", "anno_path": "Jogging/groundtruth_rect.1.txt",
        "object_class": "person"},
    {"name": "Jogging_2", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg", "anno_path": "Jogging/groundtruth_rect.2.txt",
        "object_class": "person"},
    {"name": "Jump", "path": "Jump/img", "startFrame": 1, "endFrame": 122, "nz": 4, "ext": "jpg", "anno_path": "Jump/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Jumping", "path": "Jumping/img", "startFrame": 1, "endFrame": 313, "nz": 4, "ext": "jpg", "anno_path": "Jumping/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "KiteSurf", "path": "KiteSurf/img", "startFrame": 1, "endFrame": 84, "nz": 4, "ext": "jpg", "anno_path": "KiteSurf/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Lemming", "path": "Lemming/img", "startFrame": 1, "endFrame": 1336, "nz": 4, "ext": "jpg", "anno_path": "Lemming/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Liquor", "path": "Liquor/img", "startFrame": 1, "endFrame": 1741, "nz": 4, "ext": "jpg", "anno_path": "Liquor/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Man", "path": "Man/img", "startFrame": 1, "endFrame": 134, "nz": 4, "ext": "jpg", "anno_path": "Man/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Matrix", "path": "Matrix/img", "startFrame": 1, "endFrame": 100, "nz": 4, "ext": "jpg", "anno_path": "Matrix/groundtruth_rect.txt",
        "object_class": "person head"},
    {"name": "Mhyang", "path": "Mhyang/img", "startFrame": 1, "endFrame": 1490, "nz": 4, "ext": "jpg", "anno_path": "Mhyang/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "MotorRolling", "path": "MotorRolling/img", "startFrame": 1, "endFrame": 164, "nz": 4, "ext": "jpg", "anno_path": "MotorRolling/groundtruth_rect.txt",
        "object_class": "vehicle"},
    {"name": "MountainBike", "path": "MountainBike/img", "startFrame": 1, "endFrame": 228, "nz": 4, "ext": "jpg", "anno_path": "MountainBike/groundtruth_rect.txt",
        "object_class": "bicycle"},
    {"name": "Panda", "path": "Panda/img", "startFrame": 1, "endFrame": 1000, "nz": 4, "ext": "jpg", "anno_path": "Panda/groundtruth_rect.txt",
        "object_class": "mammal"},
    {"name": "RedTeam", "path": "RedTeam/img", "startFrame": 1, "endFrame": 1918, "nz": 4, "ext": "jpg", "anno_path": "RedTeam/groundtruth_rect.txt",
        "object_class": "vehicle"},
    {"name": "Rubik", "path": "Rubik/img", "startFrame": 1, "endFrame": 1997, "nz": 4, "ext": "jpg", "anno_path": "Rubik/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Shaking", "path": "Shaking/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg", "anno_path": "Shaking/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Singer1", "path": "Singer1/img", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg", "anno_path": "Singer1/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Singer2", "path": "Singer2/img", "startFrame": 1, "endFrame": 366, "nz": 4, "ext": "jpg", "anno_path": "Singer2/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Skater", "path": "Skater/img", "startFrame": 1, "endFrame": 160, "nz": 4, "ext": "jpg", "anno_path": "Skater/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Skater2", "path": "Skater2/img", "startFrame": 1, "endFrame": 435, "nz": 4, "ext": "jpg", "anno_path": "Skater2/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Skating1", "path": "Skating1/img", "startFrame": 1, "endFrame": 400, "nz": 4, "ext": "jpg", "anno_path": "Skating1/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Skating2_1", "path": "Skating2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg", "anno_path": "Skating2/groundtruth_rect.1.txt",
        "object_class": "person"},
    {"name": "Skating2_2", "path": "Skating2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg", "anno_path": "Skating2/groundtruth_rect.2.txt",
        "object_class": "person"},
    {"name": "Skiing", "path": "Skiing/img", "startFrame": 1, "endFrame": 81, "nz": 4, "ext": "jpg", "anno_path": "Skiing/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Soccer", "path": "Soccer/img", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg", "anno_path": "Soccer/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Subway", "path": "Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg", "anno_path": "Subway/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Surfer", "path": "Surfer/img", "startFrame": 1, "endFrame": 376, "nz": 4, "ext": "jpg", "anno_path": "Surfer/groundtruth_rect.txt",
        "object_class": "person head"},
    {"name": "Suv", "path": "Suv/img", "startFrame": 1, "endFrame": 945, "nz": 4, "ext": "jpg", "anno_path": "Suv/groundtruth_rect.txt",
        "object_class": "car"},
    {"name": "Sylvester", "path": "Sylvester/img", "startFrame": 1, "endFrame": 1345, "nz": 4, "ext": "jpg", "anno_path": "Sylvester/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Tiger1", "path": "Tiger1/img", "startFrame": 1, "endFrame": 354, "nz": 4, "ext": "jpg", "anno_path": "Tiger1/groundtruth_rect.txt", "initOmit": 5,
        "object_class": "other"},
    {"name": "Tiger2", "path": "Tiger2/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg", "anno_path": "Tiger2/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Toy", "path": "Toy/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg", "anno_path": "Toy/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Trans", "path": "Trans/img", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg", "anno_path": "Trans/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Trellis", "path": "Trellis/img", "startFrame": 1, "endFrame": 569, "nz": 4, "ext": "jpg", "anno_path": "Trellis/groundtruth_rect.txt",
        "object_class": "face"},
    {"name": "Twinnings", "path": "Twinnings/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg", "anno_path": "Twinnings/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Vase", "path": "Vase/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg", "anno_path": "Vase/groundtruth_rect.txt",
        "object_class": "other"},
    {"name": "Walking", "path": "Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg", "anno_path": "Walking/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Walking2", "path": "Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg", "anno_path": "Walking2/groundtruth_rect.txt",
        "object_class": "person"},
    {"name": "Woman", "path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg", "anno_path": "Woman/groundtruth_rect.txt",
        "object_class": "person"}
]
