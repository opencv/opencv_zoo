import argparse

import cv2 as cv
import numpy as np

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from raft import Raft

parser = argparse.ArgumentParser(description='RAFT (https://github.com/princeton-vl/RAFT)')
parser.add_argument('--input1', '-i1', type=str,
                    help='Usage: Set input1 path to first image, omit if using camera or video.')
parser.add_argument('--input2', '-i2', type=str,
                    help='Usage: Set input2 path to second image, omit if using camera or video.')
parser.add_argument('--video', '-vid', type=str,
                    help='Usage: Set video path to desired input video, omit if using camera or two image inputs.')
parser.add_argument('--model', '-m', type=str, default='optical_flow_estimation_raft_2023aug.onnx',
                    help='Usage: Set model path, defaults to optical_flow_estimation_raft_2023aug.onnx.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save a file with results. Invalid in case of camera input.')
parser.add_argument('--visual', '-vis', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

UNKNOWN_FLOW_THRESH = 1e7

def make_color_wheel():
    """ Generate color wheel according Middlebury color code.
    
    Returns:
        Color wheel(numpy.ndarray): Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

colorwheel = make_color_wheel()

def compute_color(u, v):
    """ Compute optical flow color map
    
    Args:
        u(numpy.ndarray): Optical flow horizontal map
        v(numpy.ndarray): Optical flow vertical map
        
    Returns:
        img (numpy.ndarray): Optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """Convert flow into middlebury color code image

    Args:
        flow (np.ndarray): The computed flow map
        
    Returns:
        (np.ndarray): Image corresponding to the flow map.
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def draw_flow(flow_map, img_width, img_height):
    """Convert flow map to image

    Args:
        flow_map (np.ndarray): The computed flow map
        img_width (int): The width of the first input photo
        img_height (int): The height of the first input photo

    Returns:
        (np.ndarray): Image corresponding to the flow map.
    """
	# Convert flow to image
    flow_img = flow_to_image(flow_map)
	# Convert to BGR
    flow_img = cv.cvtColor(flow_img, cv.COLOR_RGB2BGR)
	# Resize the depth map to match the input image shape
    return cv.resize(flow_img, (img_width, img_height))


def visualize(image1, image2, flow_img):
    """
    Combine two input images with resulting flow img and display them together

    Args:
        image1 (np.ndarray): The first input image.
        imag2 (np.ndarray): The second input image.
        flow_img (np.ndarray): The output flow map drawn as an image

    Returns:
        combined_img (np.ndarray): The visualized result.
    """
    combined_img = np.hstack((image1, image2, flow_img))
    cv.namedWindow("Estimated flow", cv.WINDOW_NORMAL)
    cv.imshow("Estimated flow", combined_img)
    cv.waitKey(0)
    return combined_img


if __name__ == '__main__':
    # Instantiate RAFT
    model = Raft(modelPath=args.model)

    if args.input1 is not None and args.input2 is not None:
        # Read image
        image1 = cv.imread(args.input1)
        image2 = cv.imread(args.input2)
        img_height, img_width, img_channels = image1.shape

        # Inference
        result = model.infer(image1, image2)

        # Create flow image based on the result flow map
        flow_image = draw_flow(result, img_width, img_height)

        # Save results if save is true
        if args.save:
            print('Results saved to result.jpg\n')
            cv.imwrite('result.jpg', flow_image)

        # Visualize results in a new window
        if args.visual:
            input_output_visualization = visualize(image1, image2, flow_image)
            
            
    elif args.video is not None:
        cap = cv.VideoCapture(args.video)    
        FLOW_FRAME_OFFSET = 3 # Number of frame difference to estimate the optical flow
        
        if args.visual:
            cv.namedWindow("Estimated flow", cv.WINDOW_NORMAL)
        
        frame_list = []	
        img_array = []
        frame_num = 0
        while cap.isOpened():
            try:
                # Read frame from the video
                ret, prev_frame = cap.read()
                frame_list.append(prev_frame)
                if not ret:	
                    break
            except:
                continue

            frame_num += 1
            if frame_num <= FLOW_FRAME_OFFSET:
                continue
            else:
                frame_num = 0

            result = model.infer(frame_list[0], frame_list[-1])
            img_height, img_width, img_channels = frame_list[0].shape
            flow_img = draw_flow(result, img_width, img_height)

            alpha = 0.6
            combined_img = cv.addWeighted(frame_list[0], alpha, flow_img, (1-alpha),0)

            if args.visual:
                cv.imshow("Estimated flow", combined_img)
            img_array.append(combined_img)
            # Remove the oldest frame
            frame_list.pop(0)

            # Press key q to stop
            if cv.waitKey(1) == ord('q'):
                break
            
        cap.release()

        if args.save:
            fourcc = cv.VideoWriter_fourcc(*'mp4v') 
            height,width,layers= img_array[0].shape
            video = cv.VideoWriter('result.mp4', fourcc, 30.0, (width, height), isColor=True)
            for img in img_array:
                video.write(img)
            video.release()

        cv.destroyAllWindows()


    else: # Omit input to call default camera
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        tm = cv.TickMeter()
        while cv.waitKey(30) < 0:
            hasFrame1, frame1 = cap.read()
            hasFrame2, frame2 = cap.read()
            if not hasFrame1:
                print('First frame was not grabbed!')
                break
            
            if not hasFrame2:
                print('Second frame was not grabbed!')
                break

            # Inference
            tm.start()
            result = model.infer(frame1, frame2)
            tm.stop()
            result = draw_flow(result, w, h)

            # Draw results on the input image
            frame = visualize(frame1, frame2, result)

            tm.reset()
