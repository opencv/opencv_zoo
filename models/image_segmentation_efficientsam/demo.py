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
parser.add_argument('--model', '-m', type=str, default='image_segmentation_efficientsam_ti_2025april.onnx',
                    help='Set model path, defaults to image_segmentation_efficientsam_ti_2024april.onnx.')
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

# Global configuration
WINDOW_SIZE = (800, 600)  # Fixed window size (width, height)
MAX_POINTS = 6             # Maximum allowed points
points = []                # Store clicked coordinates (original image scale)
labels = []                # Point labels (-1: useless, 0: background, 1: foreground, 2: top-left, 3: bottom right)
backend_point = []
rectangle = False
current_img = None

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
    """Handle mouse events with coordinate conversion"""
    global points, labels, backend_point, rectangle, current_img
    orig_img = param['original_img']
    image_window = param['image_window']
    
    if event == cv.EVENT_LBUTTONDOWN:
        param['mouse_down_time'] = cv.getTickCount()
        backend_point = [x, y]

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            rectangle_change_img = current_img.copy()
            cv.rectangle(rectangle_change_img, (backend_point[0], backend_point[1]), (x, y), (255,0,0) , 2)
            cv.imshow(image_window, rectangle_change_img)
        elif len(backend_point) != 0:
            rectangle = True
        

    elif event == cv.EVENT_LBUTTONUP:
        if len(points) >= MAX_POINTS:
            print(f"Maximum points reached ({MAX_POINTS})")
            return

        if rectangle == False:
            duration = (cv.getTickCount() - param['mouse_down_time'])/cv.getTickFrequency()
            label = -1 if duration > 0.5 else 1  # Long press = background
            
            points.append([backend_point[0], backend_point[1]])
            labels.append(label)
            print(f"Added {['background','foreground','background'][label]} point {backend_point}")
        else:
            if len(points) + 1 >= MAX_POINTS:
                print(f"Points reached ({MAX_POINTS}), could not add box")
                return
            point_leftup = []
            point_rightdown = []
            if x > backend_point[0] or y > backend_point[1]:
                point_leftup.extend(backend_point)
                point_rightdown.extend([x,y])
            else:
                point_leftup.extend([x,y])
                point_rightdown.extend(backend_point)
            points.append(point_leftup)
            points.append(point_rightdown)
            print(f"Added box from {point_leftup} to {point_rightdown}")
            labels.append(2)
            labels.append(3)
            rectangle = False
        backend_point.clear()
        
        marked_img = orig_img.copy()
        top_left = None 
        for (px, py), lbl in zip(points, labels):
            if lbl == -1:
                cv.circle(marked_img, (px, py), 5, (0, 0, 255), -1)
            elif lbl == 1:
                cv.circle(marked_img, (px, py), 5, (0, 255, 0), -1)
            elif lbl == 2:
                top_left = (px, py)  
            elif lbl == 3:
                bottom_right = (px, py)  
                cv.rectangle(marked_img, top_left, bottom_right, (255,0,0) , 2)            
        cv.imshow(image_window, marked_img)
        current_img = marked_img.copy()
        

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
        image_window = "Origin image"
        cv.namedWindow(image_window, cv.WINDOW_NORMAL)
        # change window size
        rate = 1
        rate1 = 1
        rate2 = 1
        if(image.shape[1]>WINDOW_SIZE[0]):
            rate1 = WINDOW_SIZE[0]/image.shape[1]
        if(image.shape[0]>WINDOW_SIZE[1]):
            rate2 = WINDOW_SIZE[1]/image.shape[0]
        rate = min(rate1, rate2)
        # width, height
        WINDOW_SIZE = (int(image.shape[1] * rate), int(image.shape[0] * rate))
        cv.resizeWindow(image_window, WINDOW_SIZE[0], WINDOW_SIZE[1])
        # put the window on the left of the screen
        cv.moveWindow(image_window, 50, 100)
        # set listener to record user's click point
        param = {
            'original_img': image,
            'mouse_down_time': 0,
            'image_window' : image_window
        }
        cv.setMouseCallback(image_window, select, param)
        # tips in the terminal
        print("Click — Select foreground point\n"
        "Long press — Select background point\n"
        "Drag — Create selection box\n"
        "Enter — Infer\n"
        "Backspace — Clear the prompts")
        # show image
        cv.imshow(image_window, image)
        current_img = image.copy()
        # create window to show visualized result
        vis_image = image.copy()
        segmentation_window = "Segment result"
        cv.namedWindow(segmentation_window, cv.WINDOW_NORMAL)
        cv.resizeWindow(segmentation_window, WINDOW_SIZE[0], WINDOW_SIZE[1])
        cv.moveWindow(segmentation_window, WINDOW_SIZE[0]+51, 100)
        cv.imshow(segmentation_window, vis_image)
        # waiting for click
        while True:
            # Check window status
            # if click × to close the image window then ending
            if (cv.getWindowProperty(image_window, cv.WND_PROP_VISIBLE) < 1 or 
                cv.getWindowProperty(segmentation_window, cv.WND_PROP_VISIBLE) < 1):
                break
        
            # Handle keyboard input
            key = cv.waitKey(1)
            
            # receive enter
            if key == 13:
                
                vis_image = image.copy()
                cv.putText(vis_image, "infering...", 
                            (50, vis_image.shape[0]//2), 
                            cv.FONT_HERSHEY_SIMPLEX, 10, (255,255,255), 5)
                cv.imshow(segmentation_window, vis_image)
                
                result = model.infer(image=image, points=points, labels=labels)
                if len(result) == 0:
                    print("clear and select points again!")
                else:    
                    vis_result = visualize(image, result)
                    
                    cv.imshow(segmentation_window, vis_result)
            elif key == 8:  # ASCII for Backspace
                points.clear()
                labels.clear()
                backend_point = []
                rectangle = False
                current_img = image
                print("poins clear up")
                cv.imshow(image_window, image)
                
        cv.destroyAllWindows()
        
        # Save results if save is true
        if args.save:
            cv.imwrite('./example_outputs/vis_result.jpg', vis_result)
            cv.imwrite("./example_outputs/mask.jpg", result)
            print('vis_result.jpg and mask.jpg are saved to ./example_outputs/')
        
    else:
        print('Set input path to a certain image.')
        pass
        
