import os
import numpy as np
import cv2 as cv
import tqdm


# Official CCPD test subsets (Xu et al., ECCV 2018).
# Images in CCPD-Base are train/val only and must not be scored by default.
CCPD_TEST_SUBSETS = (
    "ccpd_db",
    "ccpd_blur",
    "ccpd_fn",
    "ccpd_rotate",
    "ccpd_tilt",
    "ccpd_challenge",
)

SUBSET_DISPLAY = {
    "ccpd_db": "DB",
    "ccpd_blur": "Blur",
    "ccpd_fn": "FN",
    "ccpd_rotate": "Rotate",
    "ccpd_tilt": "Tilt",
    "ccpd_challenge": "Challenge",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def parse_ccpd_bbox(filename):
    """Parse the axis-aligned plate box from a CCPD filename.

    A sample name is ``025-95_113-154&383_386&473-...jpg``.
    Field 3 (0-based index 2) is the box ``x1&y1_x2&y2``.
    See https://github.com/detectRecog/CCPD#dataset-annotations
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("-")
    if len(parts) < 3:
        return None
    try:
        left, right = parts[2].split("_")
        x1, y1 = left.split("&")
        x2, y2 = right.split("&")
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    except ValueError:
        return None
    return np.array(
        [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        dtype=np.float32,
    )


def corners_to_xyxy(det):
    """Convert LPD-YuNet detections to axis-aligned boxes.

    ``LPD_YuNet.infer`` returns ``N x 9``:
    ``x1 y1 x2 y2 x3 y3 x4 y4 score``.
    """
    if det is None:
        return np.zeros((0, 5), dtype=np.float32)
    det = np.asarray(det, dtype=np.float32)
    if det.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if det.ndim == 1:
        det = det.reshape(1, -1)
    if det.shape[1] < 9:
        return np.zeros((0, 5), dtype=np.float32)
    pts = det[:, :8].reshape(-1, 4, 2)
    xyxy = np.concatenate(
        [pts.min(axis=1), pts.max(axis=1), det[:, -1:]],
        axis=1,
    )
    return xyxy.astype(np.float32)


def box_iou(boxes, gt):
    """IoU between ``boxes`` (N, 4) xyxy and a single ``gt`` (4,) xyxy."""
    boxes = np.asarray(boxes, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    tl = np.maximum(boxes[:, :2], gt[:2])
    br = np.minimum(boxes[:, 2:4], gt[2:4])
    wh = np.clip(br - tl, a_min=0, a_max=None)
    inter = wh[:, 0] * wh[:, 1]
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    area_gt = max(0.0, float(gt[2] - gt[0])) * max(0.0, float(gt[3] - gt[1]))
    union = area_boxes + area_gt - inter
    return inter / np.clip(union, 1e-6, None)


def voc_ap(rec, prec):
    """VOC-style AP: area under the precision envelope."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def _is_image(path):
    return os.path.splitext(path)[1] in IMAGE_EXTS


def _list_images_in_dir(directory):
    if not os.path.isdir(directory):
        return []
    files = []
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path) and _is_image(path) and parse_ccpd_bbox(path) is not None:
            files.append(path)
    return sorted(files)


class CCPD:
    """CCPD license-plate detection evaluation.

    Follows the official detection protocol from Xu et al., ECCV 2018
    (https://github.com/detectRecog/CCPD):

    * Each image contains exactly one license plate.
    * The detector may output only one box per image (highest score).
    * A box is correct if and only if IoU with the ground-truth box is
      greater than 0.7.
    * The headline metric is precision on each official test subset
      (DB, Blur, FN, Rotate, Tilt, Challenge) and on their union. The
      CCPD paper labels that union score ``AP`` in its detection table.

    When ``root`` is the extracted CCPD2019 directory, the six official
    test folders are used and ``ccpd_base`` (train/val) is ignored.
    When ``root`` is a single folder of CCPD-named images, that folder is
    evaluated as one subset.
    """

    def __init__(self, root, iou_thresh=0.7):
        self.root = root
        self.iou_thresh = float(iou_thresh)
        self.subsets = self._discover_subsets(root)
        if not self.subsets:
            raise FileNotFoundError(
                "No CCPD images found under {}. "
                "Pass the extracted CCPD2019 root (containing ccpd_db, "
                "ccpd_blur, ccpd_fn, ccpd_rotate, ccpd_tilt, ccpd_challenge) "
                "or a folder of CCPD-named .jpg files.".format(root)
            )
        self.subset_stats = {}
        self.precision = 0.0
        self.ap = 0.0
        self.n_images = 0
        self.n_correct = 0

    @property
    def name(self):
        return self.__class__.__name__

    def _discover_subsets(self, root):
        subsets = []
        found_official = False
        for key in CCPD_TEST_SUBSETS:
            paths = _list_images_in_dir(os.path.join(root, key))
            if paths:
                found_official = True
                subsets.append((key, paths))
        if found_official:
            return subsets

        # A single subset directory, or any folder of CCPD-named images.
        paths = _list_images_in_dir(root)
        if paths:
            label = os.path.basename(os.path.normpath(root)) or "custom"
            return [(label, paths)]

        # Recurse, but never score ccpd_base (train/val).
        collected = {}
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            top = rel.split(os.sep)[0]
            if top == "ccpd_base":
                dirnames[:] = []
                continue
            for name in filenames:
                path = os.path.join(dirpath, name)
                if _is_image(path) and parse_ccpd_bbox(path) is not None:
                    key = top if top != "." else "custom"
                    collected.setdefault(key, []).append(path)
        for key in CCPD_TEST_SUBSETS:
            if key in collected:
                subsets.append((key, sorted(collected.pop(key))))
        for key in sorted(collected):
            subsets.append((key, sorted(collected[key])))
        return subsets

    def eval(self, model):
        scores = []
        matches = []
        self.subset_stats = {}
        last_size = None

        for key, paths in self.subsets:
            n_ok = 0
            n_used = 0
            pbar = tqdm.tqdm(paths)
            pbar.set_description_str(
                "Evaluating {} with {} / {}".format(model.name, self.name, key)
            )
            for path in pbar:
                img = cv.imread(path)
                if img is None:
                    continue
                gt = parse_ccpd_bbox(path)
                if gt is None:
                    continue

                h, w = img.shape[:2]
                size = (w, h)
                if size != last_size:
                    model.setInputSize([w, h])
                    last_size = size

                pred = corners_to_xyxy(model.infer(img))

                # Official protocol: one box per image. No box => incorrect.
                hit = False
                score = 0.0
                if len(pred) > 0:
                    best = pred[int(np.argmax(pred[:, 4]))]
                    score = float(best[4])
                    hit = float(box_iou(best[None, :4], gt)[0]) > self.iou_thresh

                scores.append(score)
                matches.append(1.0 if hit else 0.0)
                n_used += 1
                if hit:
                    n_ok += 1

            prec = float(n_ok) / float(n_used) if n_used else 0.0
            self.subset_stats[key] = dict(n=n_used, correct=n_ok, precision=prec)

        self.n_images = int(sum(s["n"] for s in self.subset_stats.values()))
        self.n_correct = int(sum(s["correct"] for s in self.subset_stats.values()))
        self.precision = (
            float(self.n_correct) / float(self.n_images) if self.n_images else 0.0
        )

        if self.n_images == 0 or len(scores) == 0:
            self.ap = 0.0
            return

        scores = np.asarray(scores, dtype=np.float32)
        matches = np.asarray(matches, dtype=np.float32)
        order = np.argsort(-scores)
        matches = matches[order]
        tp = np.cumsum(matches)
        fp = np.cumsum(1.0 - matches)
        rec = tp / float(self.n_images)
        prec = tp / np.clip(tp + fp, 1e-6, None)
        self.ap = voc_ap(rec, prec)

    def get_result(self):
        return dict(
            precision=self.precision,
            ap=self.ap,
            n_images=self.n_images,
            n_correct=self.n_correct,
            iou_thresh=self.iou_thresh,
            subsets={k: dict(s) for k, s in self.subset_stats.items()},
        )

    def print_result(self):
        print("==================== Results ====================")
        print(
            "Dataset: CCPD  |  Protocol: top-1 IoU > {} (ECCV 2018)".format(
                self.iou_thresh
            )
        )
        print("Images: {}".format(self.n_images))
        print("")
        print("{:<14} {:>8} {:>10}".format("Subset", "Images", "Precision"))
        for key, _ in self.subsets:
            stats = self.subset_stats.get(key, dict(n=0, precision=0.0))
            label = SUBSET_DISPLAY.get(key, key)
            print(
                "{:<14} {:>8d} {:>10.4f}".format(
                    label, stats["n"], stats["precision"]
                )
            )
        print(
            "{:<14} {:>8d} {:>10.4f}".format(
                "Overall", self.n_images, self.precision
            )
        )
        print("")
        print("VOC AP @ {}: {:.4f}".format(self.iou_thresh, self.ap))
        print("=================================================")
