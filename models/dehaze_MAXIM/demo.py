#
#
#
#

# A demo on how to perform inference with the class with a single image and webcam stream (if applicable) as input.

import argparse
import os

# import cv2 as cv
#
# from maxim import MAXIM
from PIL import Image
import tensorflow as tf
import collections

import numpy as np

parser = argparse.ArgumentParser(
    description='MAXIM: Multi-Axis MLP for Image Processing (CVPR 2022 Oral) (https://github.com/google-research/maxim).')
parser.add_argument('--input', '-i', default='./examples/input', type=str, help='Path to the input image.')
parser.add_argument('--model', '-m', type=str, default='./dehazed_maxim_2022Aug.tflite', help='Path to the model.')
# parser.add_argument('--backend', '-b', type=int, default=backends[0], help=help_msg_backends.format(*backends))
parser.add_argument('--has_target', type=bool, default=False, help='Set true if has target.')
parser.add_argument('--target', '-t', default='./examples/target', type=str, help='Path to the target image.')
# parser.add_argument('--conf_threshold', type=float, default=0.9, help='Filter out faces of confidence < conf_threshold.')
# parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
# parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=bool, default=True, help='Set true to save results.')
parser.add_argument('--vis', '-v', type=bool, default=True, help='Set true to open a window for result visualization. ')
parser.add_argument('--output', '-d', type=str, default='./examples/output', help='Path to the output directory.')
parser.add_argument('--geometric_ensemble', type=bool, default=False, help='Whether use ensemble infernce.')
args = parser.parse_args()

# flags.DEFINE_enum(
#     'task', 'Denoising',
#     ['Denoising', 'Deblurring', 'Deraining', 'Dehazing', 'Enhancement'],
#     'Task to run.')
# flags.DEFINE_string('ckpt_path', '', 'Path to checkpoint.')
# flags.DEFINE_string('input_dir', '', 'Input dir to the test set.')
# flags.DEFINE_string('output_dir', '', 'Output dir to store predicted images.')
# flags.DEFINE_boolean('has_target', True, 'Whether has corresponding gt image.')
# flags.DEFINE_boolean('save_images', True, 'Dump predicted images.')
# flags.DEFINE_boolean('geometric_ensemble', False,
#                      'Whether use ensemble infernce.')


# from datetime import datetime
#
# start = datetime.now()
# print(start)

with open(args.model, 'rb') as fid:
    tflite_model = fid.read()

print("Loading interpreter....")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
print("Done!")
print("Allocating tensors....")
interpreter.allocate_tensors()
print("Done!")

"""Run evaluation."""


def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.

    This function is useful to analyze checkpoints that are saved by our programs
    without need to access the exact source code of the experiment. In particular,
    it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
    subtree of parameters.
    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.
    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if '/' not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split('/', 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def calculate_psnr(img1, img2, crop_border, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
      img1 (ndarray): Images with range [0, 255].
      img2 (ndarray): Images with range [0, 255].
      crop_border (int): Cropped pixels in each edge of an image. These
          pixels are not involved in the PSNR calculation.
      test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
      float: psnr result.
    """
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
      img (ndarray): The input image. It accepts:
          1. np.uint8 type with range [0, 255];
          2. np.float32 type with range [0, 1].
    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
      img (ndarray): The image to be converted with np.float32 type and
          range [0, 255].
      dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
          converts the image to np.uint8 type with range [0, 255]. If
          dst_type is np.float32, it converts the image to np.float32 type
          with range [0, 1].
    Returns:
      (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.

    return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
    """Convert a RGB image to YCbCr image.

    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
      img (ndarray): The input image. It accepts:
          1. np.uint8 type with range [0, 255];
          2. np.float32 type with range [0, 1].
      y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
      ndarray: The converted YCbCr image. The output image has the same type
          and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(img,
                            [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                             [24.966, 112.0, -18.214]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
      img (ndarray): Images with range [0, 255].
    Returns:
      (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = rgb2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def augment_image(image, times=8):
    """Geometric augmentation."""
    if times == 4:  # only rotate image
        images = []
        for k in range(0, 4):
            images.append(np.rot90(image, k=k))
        images = np.stack(images, axis=0)
    elif times == 8:  # roate and flip image
        images = []
        for k in range(0, 4):
            images.append(np.rot90(image, k=k))
        image = np.fliplr(image)
        for k in range(0, 4):
            images.append(np.rot90(image, k=k))
        images = np.stack(images, axis=0)
    else:
        raise Exception(f'Error times: {times}')
    return images


def deaugment_image(images, times=8):
    """Reverse the geometric augmentation."""

    if times == 4:  # only rotate image
        image = []
        for k in range(0, 4):
            image.append(np.rot90(images[k], k=4 - k))
        image = np.stack(image, axis=0)
        image = np.mean(image, axis=0)
    elif times == 8:  # roate and flip image
        image = []
        for k in range(0, 4):
            image.append(np.rot90(images[k], k=4 - k))
        for k in range(0, 4):
            image.append(np.fliplr(np.rot90(images[4 + k], k=4 - k)))
        image = np.mean(image, axis=0)
    else:
        raise Exception(f'Error times: {times}')
    return image


def is_image_file(filename):
    """Check if it is an valid image file by extension."""
    return any(
        filename.endswith(extension)
        for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def save_img(img, pth):
    """Save an image to disk.

    Args:
      img: np.ndarry, [height, width, channels], img will be clipped to [0, 1]
        before saved to pth.
      pth: string, path to save the image to.
    """
    Image.fromarray(np.array(
        (np.clip(img, 0., 1.) * 255.).astype(np.uint8))).save(pth, 'PNG')


def convert_img_shape(image, factor=64):
    """Pad the image to 512*640."""
    height, width = image.shape[0], image.shape[1]
    padh = 512 - height
    padw = 640 - width
    image = np.pad(image, [(0, padh), (0, padw), (0, 0)], mode='reflect')
    return image


def _process_file(i):
    print(f'Processing {i + 1} / {num_images}...')
    input_file = input_filenames[i]
    input_img = np.asarray(Image.open(input_file).convert('RGB'),
                           np.float32) / 255.
    if args.has_target:
        target_file = target_filenames[i]
        target_img = np.asarray(Image.open(target_file).convert('RGB'),
                                np.float32) / 255.

    # Padding images to have even shapes
    height, width = input_img.shape[0], input_img.shape[1]
    input_img = convert_img_shape(input_img)
    height_even, width_even = input_img.shape[0], input_img.shape[1]

    if args.geometric_ensemble:
        input_img = augment_image(input_img, args.ensemble_times)
    else:
        input_img = np.expand_dims(input_img, axis=0)

    # handle multi-stage outputs, obtain the last scale output of last stage
    interpreter.set_tensor(input_index, input_img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    # preds = model.apply({'params': flax.core.freeze(params)}, input_img)

    # if isinstance(preds, list):
    #   preds = preds[-1]
    #   if isinstance(preds, list):
    #     preds = preds[-1]

    # De-ensemble by averaging inferenced results.
    if args.geometric_ensemble:
        preds = deaugment_image(preds, args.ensemble_times)
    else:
        preds = np.array(preds[0], np.float32)

    # unpad images to get the original resolution
    new_height, new_width = preds.shape[0], preds.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width
    preds = preds[h_start:h_end, w_start:w_end, :]

    # print PSNR scores
    if args.has_target:
        psnr = calculate_psnr(
            target_img * 255., preds * 255., crop_border=0, test_y_channel=False)
        print(f'{i}th image: psnr = {psnr:.4f}')
    else:
        psnr = -1

    # save files
    basename = os.path.basename(input_file)
    if args.save:
        save_pth = os.path.join(args.output, basename)
        save_img(preds, save_pth)

    return psnr


if args.save:
    os.makedirs(args.output, exist_ok=True)

# sorted is important for continuning an inference job.
filepath = sorted(os.listdir(args.input))
input_filenames = [
    os.path.join(args.input, x)
    for x in filepath
    if is_image_file(x)
]

if args.has_target:
    target_filenames = [
        os.path.join(args.target, x)
        for x in filepath
        if is_image_file(x)
    ]

num_images = len(input_filenames)
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[-1]["index"]

psnr_all = []
for i in range(num_images):
    psnr = _process_file(i)
    psnr_all.append(psnr)

psnr_all = np.asarray(psnr_all)

print(f'average psnr = {np.sum(psnr_all) / num_images:.4f}')
print(f'std psnr = {np.std(psnr_all):.4f}')

# if __name__ == '__main__':
#     pass
