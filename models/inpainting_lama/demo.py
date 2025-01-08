import cv2 as cv
import numpy as np
import argparse
from lama import Lama

def get_args_parser(func_args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input', help='Path to input image', default=0, required=False)
    parser.add_argument('--model', help='Path to lama onnx', default='inpainting_lama_2025jan.onnx', required=False)

    parser = argparse.ArgumentParser(parents=[parser],
                                     description='', formatter_class=argparse.RawTextHelpFormatter)
    return parser.parse_args(func_args)

drawing = False
mask_gray = None
brush_size = 15

def draw_mask(event, x, y, flags, param):
    global drawing, mask_gray, brush_size
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.circle(mask_gray, (x, y), brush_size, (255), thickness=-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

def main(func_args=None):
    global mask_gray, brush_size
    args = get_args_parser(func_args)

    lama = Lama(modelPath=args.model)
    input_image = cv.imread(args.input)
    mask_gray = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)

    stdSize = 0.6
    stdWeight = 2
    stdImgSize = 512
    imgWidth = min(input_image.shape[:2])
    fontSize = min(1.5, (stdSize*imgWidth)/stdImgSize)
    fontThickness = max(1,(stdWeight*imgWidth)//stdImgSize)

    cv.namedWindow("Draw Mask")
    cv.setMouseCallback("Draw Mask", draw_mask)

    label = "Draw the mask on the image. Press space bar when done."
    labelSize, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, fontSize, fontThickness)
    while True:
        display_image = input_image.copy()
        overlay = input_image.copy()

        alpha = 0.5
        cv.rectangle(overlay, (0, 0), (labelSize[0]+10, labelSize[1]+int(30*fontSize)), (255, 255, 255), cv.FILLED)
        cv.addWeighted(overlay, alpha, display_image, 1 - alpha, 0, display_image)

        cv.putText(display_image, label, (10, int(25*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        cv.putText(display_image, "Press 'i' to increase and 'd' to decrease brush size.", (10, int(50*fontSize)), cv.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
        display_image[mask_gray > 0] = [255, 255, 255]
        cv.imshow("Draw Mask", display_image)

        key = cv.waitKey(1) & 0xFF
        if key == ord('i'):  # Increase brush size
            brush_size += 1
            print(f"Brush size increased to {brush_size}")
        elif key == ord('d'):  # Decrease brush size
            brush_size = max(1, brush_size - 1)
            print(f"Brush size decreased to {brush_size}")
        elif key == ord(' '): # Press space bar to finish drawing
            break
        elif key == 27:
            exit()
    cv.destroyAllWindows()

    tm = cv.TickMeter()
    tm.start()
    result = lama.infer(input_image, mask_gray)
    tm.stop()
    label = 'Inference time: {:.2f} ms'.format(tm.getTimeMilli())
    cv.putText(result, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))

    cv.imshow("Inpainted Output", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
