import argparse
import numpy as np
import cv2 as cv
from efficientSAM import EfficientSAM

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='EfficientSAM Demo')
parser.add_argument('--input', '-i', type=str,
                    help='Set input path to a certain image.')
parser.add_argument('--model', '-m', type=str, default='image_segmentation_efficientsam_ti_2024may.onnx',
                    help='Set model path, defaults to image_segmentation_efficientsam_ti_2024may.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--save', '-s', action='store_true',
                    help='Specify to save a file with results. Invalid in case of camera input.')
args = parser.parse_args()

#global click listener
clicked_left = False
#global point record in the window
point = []

def visualize(image, result):
    """
    Visualize the inference result on the input image.

    Args:
        image (np.ndarray): The input image.
        result (np.ndarray): The inference result.

    Returns:
        vis_result (np.ndarray): The visualized result.
    """
    # get image and mask
    vis_result = np.copy(image)
    mask = np.copy(result)
    # change mask to binary image
    t, binary = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    assert set(np.unique(binary)) <= {0, 255}, "The mask must be a binary image"
    # enhance red channel to make the segmentation more obviously
    enhancement_factor = 1.8
    red_channel = vis_result[:, :, 2]  
    # update the channel
    red_channel = np.where(binary == 255, np.minimum(red_channel * enhancement_factor, 255), red_channel)
    vis_result[:, :, 2] = red_channel  
    
    # draw borders
    contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
    cv.drawContours(vis_result, contours, contourIdx = -1, color = (255,255,255), thickness=2)
    return vis_result

def select(event, x, y, flags, param):
    global clicked_left
    # When the left mouse button is pressed, record the coordinates of the point where it is pressed
    if event == cv.EVENT_LBUTTONUP:
        point.append([x,y])
        print("point:",point[0])
        clicked_left = True

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    # Load the EfficientSAM model
    model = EfficientSAM(modelPath=args.model)

    if args.input is not None:
        # Read image
        image = cv.imread(args.input)
        if image is None:
            print('Could not open or find the image:', args.input)
            exit(0)
        # create window
        image_window = "image: click on the thing whick you want to segment!"
        cv.namedWindow(image_window, cv.WINDOW_NORMAL)
        # change window size
        cv.resizeWindow(image_window, 800 if image.shape[0] > 800 else image.shape[0], 600 if image.shape[1] > 600 else image.shape[1])
        # put the window on the left of the screen
        cv.moveWindow(image_window, 50, 100)
        # set listener to record user's click point
        cv.setMouseCallback(image_window, select)
        # tips in the terminal
        print("click the picture on the LEFT and see the result on the RIGHT!")
        # show image
        cv.imshow(image_window, image)
        # waiting for click
        while cv.waitKey(1) == -1 or clicked_left:
            # receive click
            if clicked_left:
                # put the click point (x,y) into the model to predict
                result = model.infer(image=image, points=point, labels=[1])
                # get the visualized result
                vis_result = visualize(image, result)
                # create window to show visualized result
                cv.namedWindow("vis_result", cv.WINDOW_NORMAL)
                cv.resizeWindow("vis_result", 800 if vis_result.shape[0] > 800 else vis_result.shape[0], 600 if vis_result.shape[1] > 600 else vis_result.shape[1])
                cv.moveWindow("vis_result", 851, 100)
                cv.imshow("vis_result", vis_result)
                # set click false to listen another click
                clicked_left = False
            elif cv.getWindowProperty(image_window, cv.WND_PROP_VISIBLE) < 1: 
                # if click × to close the image window then ending
                break
            else:
                # when not clicked, set point to empty
                point = []
        cv.destroyAllWindows()
        
        # Save results if save is true
        if args.save:
            cv.imwrite('./example_outputs/vis_result.jpg', vis_result)
            cv.imwrite("./example_outputs/mask.jpg", result)
            print('vis_result.jpg and mask.jpg are saved to ./example_outputs/')

        
    else:
        print('Set input path to a certain image.')
        pass
        
