# eDifFIQA(T)

eDifFIQA(T) is a light-weight version of the models presented in the paper [eDifFIQA: Towards Efficient Face Image Quality Assessment based on Denoising Diffusion Probabilistic Models](https://ieeexplore.ieee.org/document/10468647), it achieves state-of-the-art results in the field of face image quality assessment.

Notes:

- The original implementation can be found [here](https://github.com/LSIbabnikz/eDifFIQA).
- The included model combines a pretrained MobileFaceNet backbone, with a quality regression head trained using the proceedure presented in the original paper.
- The model predicts quality scores of aligned face samples, where a higher predicted score corresponds to a higher quality of the input sample.


## Demo

***NOTE***: The provided demo uses [../face_detection_yunet](../face_detection_yunet) for face detection, in order to properly align the face samples, while the original implementation uses a RetinaFace(ResNet50) model, which might cause some differences between the results of the two implementations.

To try the demo run the following commands:


```shell
# Assess the quality of 'image1'
python demo.py -i /path/to/image1

# Output all the arguments of the demo
python demo.py --help
```


### Example outputs

![ediffiqaDemo](./example_outputs/demo.jpg)

The demo outputs the quality of the sample via terminal (print) and via image in __results.jpg__. 

## License

All files in this directory are licensed under [CC-BY-4.0](./LICENSE).

