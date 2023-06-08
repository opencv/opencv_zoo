# Pose estimation from MediaPipe Pose

This model estimates 33 pose keypoints and person segmentation mask per detected person from [person detector](../person_detection_mediapipe). (The image below is referenced from [MediaPipe Pose Keypoints](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#blazepose-keypoints-used-in-mediapipe-blazepose))
 
![MediaPipe Pose Landmark](examples/pose_landmarks.png)

This model is converted from TFlite to ONNX using following tools:
- TFLite model to ONNX: https://github.com/onnx/tensorflow-onnx
- simplified by [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)

**Note**:
- Visit https://github.com/google/mediapipe/blob/master/docs/solutions/models.md#pose for models of larger scale.
## Demo

Run the following commands to try the demo:
```bash
# detect on camera input
python demo.py
# detect on an image
python demo.py -i /path/to/image -v
```

### Example outputs

![webcam demo](./example_outputs/mpposeest_demo.webp)

## License

All files in this directory are licensed under [Apache 2.0 License](LICENSE).

## Reference
- MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- MediaPipe pose model and model card: https://github.com/google/mediapipe/blob/master/docs/solutions/models.md#pose
- BlazePose TFJS: https://github.com/tensorflow/tfjs-models/tree/master/pose-detection/src/blazepose_tfjs
