import cv2 as cv
import argparse

# Check OpenCV version
assert cv.__version__ >= "4.8.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

parser = argparse.ArgumentParser(
    description="VIT track opencv API")
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set path to the input video. Omit for using default camera.')
parser.add_argument('--model_path', type=str, default='vitTracker.onnx',
                    help='Usage: Set model path, defaults to vitTracker.onnx.')
args = parser.parse_args()

def visualize(image, bbox, score, isLocated, fps=None, box_color=(0, 255, 0),text_color=(0, 255, 0), fontScale = 1, fontSize = 1):
    output = image.copy()
    h, w, _ = output.shape

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 30), cv.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)

    if isLocated and score >= 0.3:
        # bbox: Tuple of length 4
        x, y, w, h = bbox
        cv.rectangle(output, (x, y), (x+w, y+h), box_color, 2)
        cv.putText(output, '{:.2f}'.format(score), (x, y+20), cv.FONT_HERSHEY_DUPLEX, fontScale, text_color, fontSize)
    else:
        text_size, baseline = cv.getTextSize('Target lost!', cv.FONT_HERSHEY_DUPLEX, fontScale, fontSize)
        text_x = int((w - text_size[0]) / 2)
        text_y = int((h - text_size[1]) / 2)
        cv.putText(output, 'Target lost!', (text_x, text_y), cv.FONT_HERSHEY_DUPLEX, fontScale, (0, 0, 255), fontSize)

    return output

if __name__ == '__main__':

    params = cv.TrackerVit_Params()
    params.net = args.model_path
    model = cv.TrackerVit_create(params)

    # Read from args.input
    _input = args.input
    if args.input is None:
        device_id = 0
        _input = device_id
    video = cv.VideoCapture(_input)

    # Select an object
    has_frame, first_frame = video.read()
    if not has_frame:
        print('No frames grabbed!')
        exit()
    first_frame_copy = first_frame.copy()
    cv.putText(first_frame_copy, "1. Drag a bounding box to track.", (0, 15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv.putText(first_frame_copy, "2. Press ENTER to confirm", (0, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    roi = cv.selectROI('vitTrack Demo', first_frame_copy)
    print("Selected ROI: {}".format(roi))

    # Init tracker with ROI
    model.init(first_frame, roi)

    # Track frame by frame
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        has_frame, frame = video.read()
        if not has_frame:
            print('End of video')
            break
        # Inference
        tm.start()
        isLocated, bbox = model.update(frame)
        score = model.getTrackingScore()
        tm.stop()
        # Visualize
        frame = visualize(frame, bbox, score, isLocated, fps=tm.getFPS())
        cv.imshow('vittrack Demo', frame)
        tm.reset()