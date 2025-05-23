import cv2 as cv
import argparse

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, [p.split('-')[0] for p in str_version.split('.')]))
assert opencv_python_version(cv.__version__) >= opencv_python_version("5.0.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from nafnet import Nafnet

def get_args_parser(func_args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', help='Path to input image.', default='example_outputs/licenseplate_motion.jpg', required=False)
    parser.add_argument('--model', help='Path to nafnet deblurring onnx model', default='deblurring_nafnet_2025may.onnx', required=False)

    args, _ = parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[parser],
                                     description='', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)

def main(func_args=None):
    args = get_args_parser(func_args)

    nafnet = Nafnet(modelPath=args.model)

    input_image = cv.imread(args.input)

    tm = cv.TickMeter()
    tm.start()
    result = nafnet.infer(input_image)
    tm.stop()
    label = 'Inference time: {:.2f} ms'.format(tm.getTimeMilli())
    cv.putText(result, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))

    cv.imshow("Input image", input_image)
    cv.imshow("Output image", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
