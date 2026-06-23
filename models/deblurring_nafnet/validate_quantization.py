# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Quantization validation utility for deblurring_nafnet
# Runs FP32 and INT8 ONNX models with identical preprocessing and reports timing, PSNR, and SSIM.

import argparse
import time
from typing import Tuple

import cv2 as cv
import numpy as np
import onnxruntime as ort


def get_args_parser(func_args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', help='Path to input image.', default='example_outputs/licenseplate_motion.jpg', required=False)
    parser.add_argument('--model_fp32', help='Path to FP32 ONNX model', default='deblurring_nafnet_2025may.onnx', required=False)
    parser.add_argument('--model_int8', help='Path to INT8 (quantized) ONNX model', default='deblurring_nafnet_2025may_int8.onnx', required=False)
    parser.add_argument('--save_outputs', action='store_true', help='Save output images next to models')

    args, _ = parser.parse_known_args(func_args)
    parser = argparse.ArgumentParser(parents=[parser], description='Validate quantization for deblurring_nafnet', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)


def preprocess(image: np.ndarray) -> np.ndarray:
    # Match nafnet.py: blobFromImage(image, 1/255, (W,H), mean=(0,0,0), swapRB=True, crop=False)
    blob = cv.dnn.blobFromImage(image, scalefactor=1.0/255.0, size=(image.shape[1], image.shape[0]), mean=(0, 0, 0), swapRB=True, crop=False)
    return blob.astype(np.float32)


def postprocess(output: np.ndarray) -> np.ndarray:
    # output expected shape: (1, C, H, W)
    result = output[0]
    result = np.transpose(result, (1, 2, 0))
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    result = cv.cvtColor(result, cv.COLOR_RGB2BGR)
    return result


def run_onnx(session: ort.InferenceSession, input_blob: np.ndarray) -> Tuple[np.ndarray, float]:
    input_name = session.get_inputs()[0].name
    tm = cv.TickMeter()
    tm.start()
    output_list = session.run(None, {input_name: input_blob})
    tm.stop()
    return output_list[0], tm.getTimeMilli()


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return cv.PSNR(img1, img2)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # Simple SSIM implementation averaged across channels
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    if img1.ndim == 2:
        img1 = img1[..., None]
    if img2.ndim == 2:
        img2 = img2[..., None]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssim_per_channel = []
    for ch in range(img1.shape[2]):
        x = img1[:, :, ch]
        y = img2[:, :, ch]
        mu_x = cv.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv.GaussianBlur(y, (11, 11), 1.5)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y

        sigma_x_sq = cv.GaussianBlur(x * x, (11, 11), 1.5) - mu_x_sq
        sigma_y_sq = cv.GaussianBlur(y * y, (11, 11), 1.5) - mu_y_sq
        sigma_xy = cv.GaussianBlur(x * y, (11, 11), 1.5) - mu_x_mu_y

        numerator = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = numerator / (denominator + 1e-12)
        ssim_per_channel.append(ssim_map.mean())

    return float(np.mean(ssim_per_channel))


def main(func_args=None):
    args = get_args_parser(func_args)

    # Load image
    image = cv.imread(args.input)
    if image is None:
        raise FileNotFoundError(f'Failed to read input image: {args.input}')

    # Preprocess
    blob = preprocess(image)

    # Create ORT sessions
    sess_options = ort.SessionOptions()
    providers = ['CPUExecutionProvider']

    fp32_sess = ort.InferenceSession(args.model_fp32, sess_options=sess_options, providers=providers)
    int8_sess = ort.InferenceSession(args.model_int8, sess_options=sess_options, providers=providers)

    # Run inference
    fp32_out, t_fp32 = run_onnx(fp32_sess, blob)
    int8_out, t_int8 = run_onnx(int8_sess, blob)

    # Postprocess
    fp32_img = postprocess(fp32_out)
    int8_img = postprocess(int8_out)

    # Metrics
    psnr = compute_psnr(fp32_img, int8_img)
    ssim = compute_ssim(fp32_img, int8_img)

    # Display
    label_fp32 = f'FP32 time: {t_fp32:.2f} ms'
    label_int8 = f'INT8 time: {t_int8:.2f} ms\nPSNR(fp32,int8): {psnr:.2f} dB\nSSIM(fp32,int8): {ssim:.4f}'

    fp32_disp = fp32_img.copy()
    int8_disp = int8_img.copy()
    cv.putText(fp32_disp, label_fp32, (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv.putText(fp32_disp, label_fp32, (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    y = 22
    for line in label_int8.split('\n'):
        cv.putText(int8_disp, line, (8, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv.putText(int8_disp, line, (8, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 22

    cv.imshow('Input', image)
    cv.imshow('Output FP32', fp32_disp)
    cv.imshow('Output INT8', int8_disp)
    cv.waitKey(0)
    cv.destroyAllWindows()

    if args.save_outputs:
        base_fp32 = args.model_fp32.rsplit('.', 1)[0]
        base_int8 = args.model_int8.rsplit('.', 1)[0]
        cv.imwrite(base_fp32 + '_output.png', fp32_img)
        cv.imwrite(base_int8 + '_output.png', int8_img)

    print('--- Quantization Validation ---')
    print(f'FP32 model: {args.model_fp32}\n  Inference time: {t_fp32:.2f} ms')
    print(f'INT8 model: {args.model_int8}\n  Inference time: {t_int8:.2f} ms')
    print(f'PSNR (FP32 vs INT8): {psnr:.2f} dB')
    print(f'SSIM (FP32 vs INT8): {ssim:.4f}')


if __name__ == '__main__':
    main()
