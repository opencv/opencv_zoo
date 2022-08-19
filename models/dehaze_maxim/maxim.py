# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.


import numpy as np
import onnxruntime as ort
from PIL import Image
from calculate_psnr import calculate_psnr

def _is_image_file(filename):
    """Check if it is a valid image file by extension."""
    return any(
        filename.endswith(extension)
        for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


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
            psnr = calculate_psnr(
                self._target_img * 255., preds * 255., crop_border=0, test_y_channel=False)
            print(f'psnr = {psnr:.4f}')
        else:
            psnr = -1

        # save files
        if self._save:
            _save_img(preds, self._output_file)

        return preds, psnr


