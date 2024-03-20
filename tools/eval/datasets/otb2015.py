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
        if not os.path.exists(traj_file):
            txt_names = {
                'FleetFace': 'fleetface.txt',
                'Jogging-1': 'jogging_1.txt',
                'Jogging-2': 'jogging_2.txt',
                'Skating2-1': 'skating2_1.txt',
                'Skating2-2': 'skating2_2.txt',
                'FaceOcc1': 'faceocc1.txt',
                'FaceOcc2': 'faceocc2.txt',
                'Human4-2': 'human4_2.txt'
            }
            txt_name = txt_names.get(self.name, self.name[0].lower() + self.name[1:] + '.txt')
            traj_file = os.path.join("OTB_results", txt_name)

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
        with open(os.path.join(root, 'OTB.json'), 'r') as f:
            meta_data = json.load(f)
        self.root = root
        self.videos = {}
        pbar = tqdm(meta_data.keys(), desc='Loading OTB', ncols=100)
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = Video(video,
                                       self.root,
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

class OTB2015:
    def __init__(self, root):
        # Go up one if directory is provided
        root = os.path.abspath(root)
        if root.endswith("OTB2015"):
            root = os.path.dirname(root)
        print(root)

        # Unzip the OTB2015.zip file
        if os.path.exists(f'{root}/OTB2015.zip'):
            os.system(f'unzip -q "{os.path.join(root, "OTB2015.zip")}" -d "{root}"')
            os.remove(f'{root}/OTB2015.zip')

        # Move the JSON label in if it's outside
        if os.path.exists(f'{root}/OTB.json'):
            os.rename(f'{root}/OTB.json', f'{root}/OTB2015/OTB.json')

        if os.path.exists(f'{root}/OTB2015'):
            original_directories = ['Jogging', 'Skating2', 'Human4']
            updated_directories = ['Jogging-1', 'Jogging-2', 'Skating2-1', 'Skating2-2', 'Human4-2', 'OTB.json']
            original_exist = all(os.path.exists(f'{root}/OTB2015/{dir}') for dir in original_directories)
            updated_exist = all(os.path.exists(f'{root}/OTB2015/{dir}') for dir in updated_directories)
            if original_exist:
                os.rename(f'{root}/OTB2015/Jogging', f'{root}/OTB2015/Jogging-1')
                os.rename(f'{root}/OTB2015/Skating2', f'{root}/OTB2015/Skating2-1')
                os.rename(f'{root}/OTB2015/Human4', f'{root}/OTB2015/Human4-2')
                os.system(f'cp -r "{root}/OTB2015/Jogging-1" "{root}/OTB2015/Jogging-2"')
                os.system(f'cp -r "{root}/OTB2015/Skating2-1" "{root}/OTB2015/Skating2-2"')
            elif not updated_exist:
                raise RuntimeError("Not all files needed for setup are present.")

        self.root = f'{root}/OTB2015'
        self.dataset = OTBDATASET(self.root)

    @property
    def name(self):
        return self.__class__.__name__

    def eval(self, model):
        for v_idx, video in enumerate(self.dataset):
            total_time = 0
            pred_bboxes = []
            scores = []
            track_times = []

            for idx, (img, gt_bbox) in enumerate(video):
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                tic = cv.getTickCount()

                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
                    model.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                else:
                    isLocated, bbox, score = model.infer(img)
                    pred_bbox = bbox
                    scores.append(score)

                pred_bboxes.append(pred_bbox)
                toc = (cv.getTickCount() - tic) / cv.getTickFrequency()
                total_time += toc
                track_times.append(toc)

            model_path = os.path.join('OTB_results')
            os.makedirs(model_path, exist_ok=True)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for bbox in pred_bboxes:
                    f.write(','.join(map(str, bbox)) + '\n')

            avg_fps = len(video) / total_time if total_time > 0 else 0
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, total_time, avg_fps))

    def print_result(self):
        benchmark = OPEBenchmark(self.dataset)
        num_cores = cpu_count()
        evaluation_results = {}
        metrics = ["success", "precision", "norm_precision"]
        for metric in metrics:
            with Pool(processes=min(num_cores, max(1, num_cores - 1))) as pool:
                for ret in tqdm(pool.imap_unordered(benchmark.evaluate, [metric], 1), desc=f'eval {metric}', total=1, ncols=100):
                    evaluation_results[metric] = ret

        benchmark.show_result(**evaluation_results, show_video_level=False)
