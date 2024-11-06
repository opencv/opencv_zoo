# RAFT
This model is originally created by Zachary Teed and Jia Deng of Princeton University. The source code for the model is at [their repository on GitHub](https://github.com/princeton-vl/RAFT), and the original [research paper](https://arxiv.org/abs/2003.12039) is published on [Arxiv](https://arxiv.org/abs/2003.12039). The model was converted to ONNX by [PINTO0309](https://github.com/PINTO0309) in his [model zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT). The ONNX model has several variations depending on the training dataset and input dimesnions. The model used in this demo is trained on Sintel dataset with input size of 360 $\times$ 480.

**Note**:
- `optical_flow_estimation_raft_2023aug_int8bq.onnx` represents the block-quantized version in int8 precision and is generated using [block_quantize.py](../../tools/quantize/block_quantize.py) with `block_size=64`.

## Demo

Run any of the following commands to try the demo:

```shell
# run on camera input
python demo.py

# run on two images and visualize result
python demo.py --input1 /path/to/image1 --input2 /path/to/image2 -vis

# run on two images and save result
python demo.py --input1 /path/to/image1 --input2 /path/to/image2 -s

# run on two images and both save and visualize result
python demo.py --input1 /path/to/image1 --input2 /path/to/image2 -s -vis

# run on one video and visualize result
python demo.py --video /path/to/video -vis

# run on one video and save result
python demo.py --video /path/to/video -s

# run on one video and both save and visualize result
python demo.py --video /path/to/video -s -vis

# get help regarding various parameters
python demo.py --help
```

While running on video, you can press q anytime to stop. The model demo runs on camera input, video input, or takes two images to compute optical flow across frames. The save and vis arguments of the shell command are only valid in the case of using video or two images as input. To run a different variation of the model, such as a model trained on a different dataset or with a different input size, refer to [RAFT ONNX in PINTO Model Zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT) to download your chosen model. And if your chosen model has different input shape from 360 $\times$ 480, **change the input shape in raft.py line 15 to the new input shape**. Then, add the model path to the --model argument of the shell command, such as in the following example commands:

```shell
# run on camera input
python demo.py --model /path/to/model
# run on two images
python demo.py --input1 /path/to/image1 --input2 /path/to/image2 --model /path/to/model
# run on video
python demo.py --video /path/to/video  --model /path/to/model
```

### Example outputs
The visualization argument displays both image inputs as well as out result.

![Visualization example](./example_outputs/vis.png)

The save argument saves the result only.

![Output example](./example_outputs/result.jpg)



## License

The original RAFT model is under [BSD-3-Clause license](./BSD-3-LICENSE.txt). <br />
The conversion of the RAFT model to the ONNX format by [PINTO0309](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT) is under [MIT License](./MITLICENSE.txt). <br />
Some of the code in demo.py and raft.py is adapted from [ibaiGorordo's repository](https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/tree/main) under [BSD-3-Clause license](./BSD-3-LICENSE.txt).<br />

## Reference

- https://arxiv.org/abs/2003.12039
- https://github.com/princeton-vl/RAFT
- https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation/tree/main
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/252_RAFT
