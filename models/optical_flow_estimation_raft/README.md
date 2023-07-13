# RAFT
This model is originally created by Zachary Teed and Jia Deng of Princeton University. The source code for the model is at [Raft](https://github.com/princeton-vl/RAFT), and the original research paper is at [Raft](https://arxiv.org/abs/2003.12039). The model was converted to ONNX by [PINTO0309](https://github.com/PINTO0309) in his [model zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT). The ONNX model has several variations depending on the training dataset and input dimesnions. The model used is trained on KITTI dataset with input size of 240 $\times$ 320.


## Demo

Run the following command to try the demo:

```shell
# run on camera input
python demo.py
# run on two images
python demo.py --input1 /path/to/image1 --input2 /path/to/image2 -v

# get help regarding various parameters
python demo.py --help
```

The model either runs on camera input or takes two images to compute optical flow across frames. The save and vis arguments of the shell command are only valid in the case of using two images as input. To run a different variation of the model, such as a model trained on different dataset or with different input size, refer to [RAFT ONNX](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT) to download your chosen model and run the following command:

```shell
# run on camera input
python demo.py --model /path/to/model
# run on an image
python demo.py --input1 /path/to/image1 --input2 /path/to/image2 -v --model /path/to/model
```

### Example outputs
The visualization argument displays both image inputs as well as out result.

![Visualization example](./example_outputs/vis.png)

The save argument saves the result only.

![Output example](./example_outputs/result.jpg)



## License

The original RAFT model is under [BSD-3-Clause license](./BSD-3-LICENSE.txt).
The conversion of the RAFT model to the ONNX format by [PINTO0309](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT) is under M[MIT License](./MITLICENSE.txt).
Some of the code in demo.py and raft.py is adapted from [ibaiGorordo's repository](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/tree/main) under [BSD-3-Clause license](./BSD-3-LICENSE.txt).

## Reference

- https://arxiv.org/abs/2003.12039
- https://github.com/princeton-vl/RAFT
- https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/tree/main
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT
