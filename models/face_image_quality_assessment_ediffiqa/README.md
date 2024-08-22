# eDifFIQA(T)

eDifFIQA(T) is a light-weight version of the models presented in the paper [eDifFIQA: Towards Efficient Face Image Quality Assessment based on Denoising Diffusion Probabilistic Models](https://ieeexplore.ieee.org/document/10468647), it achieves state-of-the-art results in the field of face image quality assessment.

Notes:

- The original implementation can be found [here](https://github.com/LSIbabnikz/eDifFIQA).
- The included model combines a pretrained MobileFaceNet backbone, with a quality regression head trained using the proceedure presented in the original paper.
- The model predicts quality scores of aligned face samples, where a higher predicted score corresponds to a higher quality of the input sample.

- In the figure below we show the quality distribution on two distinct datasets: LFW[[1]](#1) and XQLFW[[2]](#2). The LFW dataset contains images of relatively high quality, whereas the XQLFW dataset contains images of variable quality. There is a clear difference between the two distributions, with high quality images from the LFW dataset receiving quality scores higher than 0.5, while the mixed images from XQLFW receive much lower quality scores on average.


![qualityDist](./quality_distribution.png)


<a id="1">[1]</a> 
B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller 
“Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments” 
University of Massachusetts, Amherst, Tech. Rep. 07-49,
October 2007.

<a id="2">[2]</a> 
M. Knoche, S. Hormann, and G. Rigoll
“Cross-Quality LFW: A Database for Analyzing Cross-Resolution Image Face Recognition in Unconstrained Environments,” in Proceedings of the IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2021, pp. 1–5.



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

