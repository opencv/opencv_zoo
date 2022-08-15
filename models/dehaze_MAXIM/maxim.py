# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.


import numpy as np
import onnxruntime as ort
from PIL import Image

def _is_image_file(filename):
    """Check if it is a valid image file by extension."""
    return any(
        filename.endswith(extension)
        for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.
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
    It is mainly used for post-processing images in colorspace conversion
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


def _rgb2ycbcr(img, y_only=False):
    """Convert an RGB image to YCbCr image.

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


def _to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
    img (ndarray): Images with range [0, 255].
    Returns:
    (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = _rgb2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _augment_image(image, times=8):
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


def _deaugment_image(images, times=8):
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


def _save_img(img, pth):
    """Save an image to disk.

    Args:
    img: np.ndarry, [height, width, channels], img will be clipped to [0, 1]
        before saved to pth.
    pth: string, path to save the image to.
    """
    Image.fromarray(np.array(
        (np.clip(img, 0., 1.) * 255.).astype(np.uint8))).save(pth, 'PNG')


def _convert_img_shape(image):
    """Pad the image to 512*640."""
    height, width = image.shape[0], image.shape[1]
    padh = 512 - height
    padw = 640 - width
    image = np.pad(image, [(0, padh), (0, padw), (0, 0)], mode='reflect')
    return image


class MAXIM:
    def __init__(self, modelPath, has_target, input_file, target_filename, geometric_ensemble, save_img, output_file):
        self._modelPath = modelPath
        self._create_maxim_model()
        self._has_target = has_target
        self._input_file = input_file
        self._target_filename = target_filename
        self._geometric_ensemble = geometric_ensemble
        self._save = save_img
        self._output_file = output_file

    @property
    def name(self):
        return self.__class__.__name__

    def _create_maxim_model(self):
        print("Loading model....")
        self.model = ort.InferenceSession(self._modelPath)
        self.input_name = self.model.get_inputs()[0].name
        self.label_name = self.model.get_outputs()[-1].name
        print("Done!")

    def _preprocess(self):

        if not _is_image_file(self._input_file):
            raise ValueError('Input file is not an image file.')

        input_file = self._input_file
        input_img = np.asarray(Image.open(input_file).convert('RGB'),
                               np.float32) / 255.
        if self._has_target:
            target_file = self._target_filename
            if not _is_image_file(target_file):
                raise ValueError('Target file is not an image file.')

            self.target_img = np.asarray(Image.open(target_file).convert('RGB'),
                                         np.float32) / 255.

        # Padding images to have even shapes
        height, width = input_img.shape[0], input_img.shape[1]

        if height > 512 or width > 640:
            raise ValueError('Input image is too large, max size is 512x640.')

        input_img = _convert_img_shape(input_img)

        if self._geometric_ensemble:
            input_img = _augment_image(input_img, self._ensemble_times)
        else:
            input_img = np.expand_dims(input_img, axis=0)

        return input_img, height, width

    def infer(self):
        # Preprocess
        input_img, height, width = self._preprocess()

        # Forward
        preds = self.model.run([self.label_name], {self.input_name: input_img})[0]

        # Postprocess
        results, psnr = self._postprocess(preds, height, width)

        return results, psnr

    def _postprocess(self, preds, height, width):
        # De-ensemble by averaging inferences results.
        if self._geometric_ensemble:
            preds = _deaugment_image(preds, self._ensemble_times)
        else:
            preds = np.array(preds[0], np.float32)

        # unpad images to get the original resolution
        new_height, new_width = preds.shape[0], preds.shape[1]
        h_start = new_height // 2 - 512 // 2
        h_end = h_start + height
        w_start = new_width // 2 - 640 // 2
        w_end = w_start + width
        preds = preds[h_start:h_end, w_start:w_end, :]

        # print PSNR scores
        if self._has_target:
            psnr = self._calculate_psnr(
                self._target_img * 255., preds * 255., crop_border=0, test_y_channel=False)
            print(f'psnr = {psnr:.4f}')
        else:
            psnr = -1

        # save files
        if self._save:
            _save_img(preds, self._output_file)

        return preds, psnr

    def _calculate_psnr(self, img1, img2, crop_border, test_y_channel=False):
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
            img1 = _to_y_channel(img1)
            img2 = _to_y_channel(img2)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20. * np.log10(255. / np.sqrt(mse))
