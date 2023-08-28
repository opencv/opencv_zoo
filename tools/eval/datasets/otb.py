import os
import json
import numpy as np
import cv2 as cv
from colorama import Style, Fore
from tqdm import tqdm
from multiprocessing import Pool

def overlap_ratio(rect1, rect2):
    '''Compute overlap ratio between two rects
    Args
        rect:2d array of N x [x,y,w,h]
    Return:
        iou
    '''
    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou
def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0, 1.05, 0.05)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    # mask = np.sum(gt_bb > 0, axis=1) == 4 #TODO check all dataset
    mask = np.sum(gt_bb[:, 2:] > 0, axis=1) == 2
    # print(len(gt_bb))
    # print(len(result_bb))
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou > thresholds_overlap[i]) / float(n_frame)
    return success

def success_error(gt_center, result_center, thresholds, n_frame):
    # n_frame = len(gt_center)
    success = np.zeros(len(thresholds))
    dist = np.ones(len(gt_center)) * (-1)
    mask = np.sum(gt_center > 0, axis=1) == 2
    dist[mask] = np.sqrt(np.sum(
        np.power(gt_center[mask] - result_center[mask], 2), axis=1))
    for i in range(len(thresholds)):
        success[i] = np.sum(dist <= thresholds[i]) / float(n_frame)
    return success

class OPEBenchmark:
    def __init__(self, dataset):
        self.dataset = dataset

    def convert_bb_to_center(self, bboxes):
        return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                         (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T

    def convert_bb_to_norm_center(self, bboxes, gt_wh):
        return self.convert_bb_to_center(bboxes) / (gt_wh+1e-16)

    def eval_success(self,tracker):
        success_ret = {}
        success_ret_ = {}
        for video in self.dataset:
            gt_traj = np.array(video.gt_traj)
            tracker_traj = video.load_tracker()
            tracker_traj = np.array(tracker_traj)
            n_frame = len(gt_traj)
            if hasattr(video, 'absent'):
                gt_traj = gt_traj[video.absent == 1]
                tracker_traj = tracker_traj[video.absent == 1]
            success_ret_[video.name] = success_overlap(gt_traj, tracker_traj, n_frame)
        success_ret["tracker"] = success_ret_
        return success_ret

    def eval_precision(self,tracker):
        precision_ret = {}
        precision_ret_ = {}
        for video in self.dataset:
            gt_traj = np.array(video.gt_traj)
            tracker_traj = video.load_tracker()
            tracker_traj = np.array(tracker_traj)
            n_frame = len(gt_traj)
            if hasattr(video, 'absent'):
                gt_traj = gt_traj[video.absent == 1]
                tracker_traj = tracker_traj[video.absent == 1]
            gt_center = self.convert_bb_to_center(gt_traj)
            tracker_center = self.convert_bb_to_center(tracker_traj)
            thresholds = np.arange(0, 51, 1)
            precision_ret_[video.name] = success_error(gt_center, tracker_center,
                    thresholds, n_frame)
        precision_ret["tracker"] = precision_ret_
        return precision_ret

    def eval_norm_precision(self,tracker):
        norm_precision_ret = {}
        norm_precision_ret_ = {}
        for video in self.dataset:
            gt_traj = np.array(video.gt_traj)
            tracker_traj = video.load_tracker()
            tracker_traj = np.array(tracker_traj)
            n_frame = len(gt_traj)
            if hasattr(video, 'absent'):
                gt_traj = gt_traj[video.absent == 1]
                tracker_traj = tracker_traj[video.absent == 1]
            gt_center_norm = self.convert_bb_to_norm_center(gt_traj, gt_traj[:, 2:4])
            tracker_center_norm = self.convert_bb_to_norm_center(tracker_traj, gt_traj[:, 2:4])
            thresholds = np.arange(0, 51, 1) / 100
            norm_precision_ret_[video.name] = success_error(gt_center_norm,
                    tracker_center_norm, thresholds, n_frame)
        norm_precision_ret["tracker"] = norm_precision_ret_
        return norm_precision_ret

    def show_result(self, success_ret, precision_ret=None,
            norm_precision_ret=None, show_video_level=False, helight_threshold=0.6):
        """pretty print result
        Args:
            result: returned dict from function eval
        """
        # sort tracker
        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc
        tracker_auc_ = sorted(tracker_auc.items(),
                             key=lambda x:x[1],
                             reverse=True)[:20]
        tracker_names = [x[0] for x in tracker_auc_]


        tracker_name_len = max((max([len(x) for x in success_ret.keys()])+2), 12)
        header = ("|{:^"+str(tracker_name_len)+"}|{:^9}|{:^16}|{:^11}|").format(
                "Tracker name", "Success", "Norm Precision", "Precision")
        formatter = "|{:^"+str(tracker_name_len)+"}|{:^9.3f}|{:^16.3f}|{:^11.3f}|"
        print('-'*len(header))
        print(header)
        print('-'*len(header))
        for tracker_name in tracker_names:
            # success = np.mean(list(success_ret[tracker_name].values()))
            success = tracker_auc[tracker_name]
            if precision_ret is not None:
                precision = np.mean(list(precision_ret[tracker_name].values()), axis=0)[20]
            else:
                precision = 0
            if norm_precision_ret is not None:
                norm_precision = np.mean(list(norm_precision_ret[tracker_name].values()),
                        axis=0)[20]
            else:
                norm_precision = 0
            print(formatter.format(tracker_name, success, norm_precision, precision))
        print('-'*len(header))

        if show_video_level and len(success_ret) < 10 \
                and precision_ret is not None \
                and len(precision_ret) < 10:
            print("\n\n")
            header1 = "|{:^21}|".format("Tracker name")
            header2 = "|{:^21}|".format("Video name")
            for tracker_name in success_ret.keys():
                # col_len = max(20, len(tracker_name))
                header1 += ("{:^21}|").format(tracker_name)
                header2 += "{:^9}|{:^11}|".format("success", "precision")
            print('-'*len(header1))
            print(header1)
            print('-'*len(header1))
            print(header2)
            print('-'*len(header1))
            videos = list(success_ret[tracker_name].keys())
            for video in videos:
                row = "|{:^21}|".format(video)
                for tracker_name in success_ret.keys():
                    success = np.mean(success_ret[tracker_name][video])
                    precision = np.mean(precision_ret[tracker_name][video])
                    success_str = "{:^9.3f}".format(success)
                    if success < helight_threshold:
                        row += f'{Fore.RED}{success_str}{Style.RESET_ALL}|'
                    else:
                        row += success_str+'|'
                    precision_str = "{:^11.3f}".format(precision)
                    if precision < helight_threshold:
                        row += f'{Fore.RED}{precision_str}{Style.RESET_ALL}|'
                    else:
                        row += precision_str+'|'
                print(row)
            print('-'*len(header1))

class Video(object):
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr):
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
        traj_file = os.path.join("OTB_results", self.name+'.txt')
        if not os.path.exists(traj_file):
            if self.name == 'FleetFace':
                txt_name = 'fleetface.txt'
            elif self.name == 'Jogging-1':
                txt_name = 'jogging_1.txt'
            elif self.name == 'Jogging-2':
                txt_name = 'jogging_2.txt'
            elif self.name == 'Skating2-1':
                txt_name = 'skating2_1.txt'
            elif self.name == 'Skating2-2':
                txt_name = 'skating2_2.txt'
            elif self.name == 'FaceOcc1':
                txt_name = 'faceocc1.txt'
            elif self.name == 'FaceOcc2':
                txt_name = 'faceocc2.txt'
            elif self.name == 'Human4-2':
                txt_name = 'human4_2.txt'
            else:
                txt_name = self.name[0].lower()+self.name[1:]+'.txt'
            traj_file = os.path.join("OTB_results", txt_name)
        if os.path.exists(traj_file):
            with open(traj_file, 'r') as f :
                pred_traj = [list(map(float, x.strip().split(',')))
                        for x in f.readlines()]
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
        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading OTB', ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = Video(video,
                                          self.root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'])
        # set attr
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        keys = sorted(list(self.videos.keys()))
        for key in keys:
            yield self.videos[key]


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h

class OTB:

    def __init__(self, root):
        self.root = root
        self.dataset = OTBDATASET(root)
    @property
    def name(self):
        return self.__class__.__name__

    def eval(self, model):
        for v_idx, video in enumerate(self.dataset):
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                # convert bgr to rgb
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                tic = cv.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
                    model.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(pred_bbox)
                    scores.append(None)
                else:
                    isLocated, bbox, score = model.infer(img)
                    pred_bbox = bbox
                    pred_bboxes.append(pred_bbox)
                    scores.append(score)
                toc += cv.getTickCount() - tic
                track_times.append((cv.getTickCount() - tic) / cv.getTickFrequency())
                if idx == 0:
                    cv.destroyAllWindows()
            toc /= cv.getTickFrequency()
            model_path = os.path.join('OTB_results')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path,'{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))


    def get_result(self):
        return self.top1_acc, self.top5_acc

    def print_result(self):
        benchmark = OPEBenchmark(self.dataset)
        success_ret = {}
        with Pool(processes=1) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,"tracker"), desc='eval success', total=1, ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=1) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,"tracker"), desc='eval precision', total=1, ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                              show_video_level=False)
