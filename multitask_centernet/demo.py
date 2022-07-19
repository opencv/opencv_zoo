from data import MultiCOCO
from models import MulCenternet
import argparse
import torch
import torch.nn as nn
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Test single image.')
    parser.add_argument('--image-path', type=str, default='../datasets/coco2017/val2017/000000000785.jpg')
    parser.add_argument('--save-root', type=str, default='./inference/')
    parser.add_argument('--checkpoint_path', type=str, default='./exps/epoch_3.pkl')
    return parser.parse_args()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    os.makedirs(args.save_root, exist_ok=True)
    image_name = args.image_path.rsplit('/', 2)[-1]
    image_path = os.path.join(args.save_root, image_name)

    transforms = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278]),
        ToTensorV2()
    ])
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = transforms(image=image)
    model = torch.load(args.checkpoint_path)
    if torch.cuda.is_available():
        model = model.cuda()
        image = augmented['image'].cuda().unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        for k, v in output.items():
            print(k, v.shape, v.dtype)

    detection = ctdet_decode(
        output["heatmap"].sigmoid_(),
        output["width_height"],
        reg=output["bbox_offset"],
    )
    detection = detection.cpu().detach().squeeze()

    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = A.Compose([
        A.Resize(512, 512)
    ])
    image = t(image=image)['image']

    for d in detection:
        x1, y1, x2, y2, score, cls = tuple(d)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image)
    print('Image has been saved to ', image_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
