import argparse
from maxim import MAXIM
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser(
    description='MAXIM: Multi-Axis MLP for Image Processing (CVPR 2022 Oral) (https://github.com/google-research/maxim).')
parser.add_argument('--input', '-i', default='./examples/input/image1.png', type=str, help='Path to the input image.')
parser.add_argument('--model', '-m', type=str, default='./dehaze_maxim_2022aug.onnx', help='Path to the model.')
parser.add_argument('--has_target', type=bool, default=False, help='Set true if has target.')
parser.add_argument('--target', '-t', default='./examples/target/image1.png', type=str,
                    help='Path to the target image.')
parser.add_argument('--save', '-s', type=bool, default=True, help='Set true to save results.')
parser.add_argument('--vis', '-v', type=bool, default=True, help='Set true to open a window for result visualization. ')
parser.add_argument('--output', '-d', type=str, default='./examples/output/dehazed_image1.png',
                    help='Path to the output directory.')
parser.add_argument('--geometric_ensemble', type=bool, default=False, help='Whether use ensemble infernce.')
args = parser.parse_args()

if __name__ == '__main__':
    model = MAXIM(modelPath=args.model,
                  has_target=args.has_target,
                  input_file=args.input,
                  target_filename=args.target,
                  geometric_ensemble=args.geometric_ensemble,
                  save_img=args.save,
                  output_file=args.output)

    print("Processing image...")
    result, psnr = model.infer()
    print("Done!")
    if args.vis:
        #display result
        cv.imshow("result", np.array(
             (np.clip(result, 0., 1.) * 255.).astype(np.uint8)))
        cv.waitKey(0)
        cv.destroyAllWindows()

    print(f'psnr = ', psnr)
