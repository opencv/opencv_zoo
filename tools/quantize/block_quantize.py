import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import helper

BITS_TO_NUMPY_TYPE = {8: np.uint8, 16: np.uint16}

CONVOLUTION = "Conv"

KERNEL_SHAPE = "kernel_shape"
STRIDE = "strides"
PADS = "pads"
DILATIONS = "dilations"
GROUP = "group"

ONNX_OPSET = 21


@dataclass
class BlockQuantizeConfig:
    input_model_path: str
    output_model_path: str
    block_size: int
    bits: int


@dataclass
class BlockQuantizeResult:
    quantized_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    scales: np.ndarray = field(default_factory=lambda: np.array([]))
    zero_point: np.ndarray = field(default_factory=lambda: np.array([]))
    block_size: int = 1
    axis: int = 1
    original_shape: Tuple = field(default_factory=tuple)
    quantization_error: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ConvParams:
    kernel_shape: List[int] = field(default_factory=list)
    strides: List[int] = field(default_factory=list)
    pads: List[int] = field(default_factory=list)
    dilations: List[int] = field(default_factory=list)
    group: int = 1
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    bias: Optional[np.ndarray] = None


def closest_divisor(number: int, divisor: int) -> int:
    for d in range(divisor, 0, -1):
        if number % d == 0:
            return d
    return 1


def block_dequantize_tensor(
    x: np.ndarray, block_axis: int, scale: np.ndarray, zero_point: np.ndarray
) -> np.ndarray:
    repeats = x.shape[block_axis] // scale.shape[block_axis]

    x_scale_elementwise = np.repeat(scale, repeats=repeats, axis=block_axis)
    x_zero_point_elementwise = np.repeat(zero_point, repeats=repeats, axis=block_axis)

    y = (
        x.astype(np.float32) - x_zero_point_elementwise.astype(np.float32)
    ) * x_scale_elementwise

    return y


def block_quantize_tensor(
    x: np.ndarray,
    block_axis: int,
    scale: np.ndarray,
    zero_point: np.ndarray,
    n_bits: int,
) -> np.ndarray:
    repeats = x.shape[block_axis] // scale.shape[block_axis]

    y_scale_elementwise = np.repeat(scale, repeats=repeats, axis=block_axis)
    y_zero_point_elementwise = np.repeat(zero_point, repeats=repeats, axis=block_axis)

    y = np.rint(x / y_scale_elementwise + y_zero_point_elementwise).astype(
        BITS_TO_NUMPY_TYPE[n_bits]
    )

    return y


def create_dequantize_node(
    node_name,
    quantized_weights,
    scales,
    zero_point,
    dequantized_weights,
    block_size,
    axis,
) -> onnx.NodeProto:
    block_size_attr = helper.make_attribute("block_size", block_size)
    axis_attr = helper.make_attribute("axis", axis)

    n = helper.make_node(
        "DequantizeLinear",
        inputs=[quantized_weights, scales, zero_point],
        outputs=[dequantized_weights],
        name=node_name,
    )
    n.attribute.extend([block_size_attr, axis_attr])
    return n


def create_reshape_node(
    node_name, dequantized_weights, shape_tensor, reshaped_weights_name
) -> onnx.NodeProto:
    return helper.make_node(
        "Reshape",
        inputs=[dequantized_weights, shape_tensor],
        outputs=[reshaped_weights_name],
        name=node_name,
    )


class BlockQuantizer:
    def __init__(self, conf: BlockQuantizeConfig) -> None:
        self.conf = conf
        self.validate_conf()

        self.model = onnx.load(conf.input_model_path)

        if self.model.opset_import[0].version != ONNX_OPSET:
            self.model = onnx.version_converter.convert_version(self.model, ONNX_OPSET)

        self.graph = self.model.graph
        self.initializers_map = {
            init.name: init for init in self.model.graph.initializer
        }

    def validate_conf(self):
        if not os.path.isfile(self.conf.input_model_path):
            raise ValueError(
                f"Input model path '{self.conf.input_model_path}' does not exist or is not a file."
            )

        if not self.conf.input_model_path.lower().endswith(".onnx"):
            raise ValueError(
                f"Input model path '{self.conf.input_model_path}' must have a .onnx extension."
            )

        if not self.conf.output_model_path.lower().endswith(".onnx"):
            raise ValueError(
                f"Output model path '{self.conf.output_model_path}' must have a .onnx extension."
            )

        if self.conf.block_size <= 0:
            raise ValueError("Block size must be a positive integer.")

        if self.conf.bits not in BITS_TO_NUMPY_TYPE:
            allowed_values = ", ".join([str(k) for k in BITS_TO_NUMPY_TYPE.keys()])
            raise ValueError(
                f"Bits must be one of the following values: [{allowed_values}]."
            )

    def get_initializer_tensor(self, name: str) -> Optional[np.ndarray]:
        if name in self.initializers_map:
            return onnx.numpy_helper.to_array(self.initializers_map[name])

        return None

    def get_conv_params(self, node: onnx.NodeProto) -> ConvParams:
        params = ConvParams()

        for attr in node.attribute:
            if attr.name == KERNEL_SHAPE:
                params.kernel_shape = onnx.helper.get_attribute_value(attr)
            elif attr.name == STRIDE:
                params.strides = onnx.helper.get_attribute_value(attr)
            elif attr.name == PADS:
                params.pads = onnx.helper.get_attribute_value(attr)
            elif attr.name == DILATIONS:
                params.dilations = onnx.helper.get_attribute_value(attr)
            elif attr.name == GROUP:
                params.group = onnx.helper.get_attribute_value(attr)

        weights_name = node.input[1]
        params.weights = self.get_initializer_tensor(weights_name)

        if len(node.input) > 2:
            bias_name = node.input[2]
            params.bias = self.get_initializer_tensor(bias_name)

        return params

    def compute_scale_zeropoint(
        self, b_min: np.ndarray, b_max: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert (
            b_min < b_max
        ).all(), (
            "minimum must be lower than maximum when computing scale and zero point"
        )

        # zero must be present in the range, this enforces qmin <= zero_point <= qmax
        b_min = np.minimum(b_min, np.zeros_like(b_min, dtype=b_min.dtype))
        b_max = np.maximum(b_max, np.zeros_like(b_max, dtype=b_max.dtype))

        qmin = np.iinfo(BITS_TO_NUMPY_TYPE[self.conf.bits]).min
        qmax = np.iinfo(BITS_TO_NUMPY_TYPE[self.conf.bits]).max

        dq = qmax - qmin

        scales = (b_max - b_min) / dq
        zeropoints = np.rint(qmin - b_min / scales).astype(
            BITS_TO_NUMPY_TYPE[self.conf.bits]
        )

        return (scales, zeropoints)

    def block_quantize(self, weight: np.ndarray) -> BlockQuantizeResult:
        original_shape = weight.shape
        weight = weight.reshape((weight.shape[0], -1))

        quantization_axis = 1

        block_size = closest_divisor(weight.shape[1], self.conf.block_size)

        assert (
            weight.shape[1] % block_size == 0
        ), f"weight shape ({weight.shape[1]}) must be divisible by block size ({block_size})"

        # Warning, axis = 1 specific instruction!
        blocked_weight = weight.reshape(
            (weight.shape[0], weight.shape[1] // block_size, -1)
        )

        # Warning, axis = 1 specific instruction!
        blocked_max = np.max(blocked_weight, -1)
        # Warning, axis = 1 specific instruction!
        blocked_min = np.min(blocked_weight, -1)

        scales, zeropoints = self.compute_scale_zeropoint(blocked_min, blocked_max)

        quantized_weight = block_quantize_tensor(
            weight, quantization_axis, scales, zeropoints, self.conf.bits
        )
        reconstructed_mat = block_dequantize_tensor(
            quantized_weight, quantization_axis, scales, zeropoints
        )

        qerror = np.linalg.norm(reconstructed_mat - weight)

        res = BlockQuantizeResult(
            quantized_weight,
            scales,
            zeropoints,
            block_size,
            quantization_axis,
            original_shape,
            qerror,
        )

        return res

    def get_model_size(self, model_path: str) -> float:
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / 1024

        return size_mb

    def display_summary(self, sqe: List):
        mse = sum(sqe) / len(sqe)
        original_model_size = self.get_model_size(self.conf.input_model_path)
        quantized_model_size = self.get_model_size(self.conf.output_model_path)

        print("Done! Results saved in", self.conf.output_model_path)
        print("\nSummary of Results:\n")
        print(f"{'Metric':<30} {'Value':<10}")
        print(f"{'-'*40}")
        print(f"{'Mean Squared Quantization Error':<30} {mse:.6f}")
        print(f"{'Original Model Size (KB)':<31} {original_model_size:,.2f}")
        print(f"{'Block-Quantized Model Size (KB)':<30} {quantized_model_size:,.2f}")

    def run(self):
        print("Quantizing the model...")

        visited_nodes = []
        sqe = []

        for node in self.model.graph.node:
            if node.name in visited_nodes:
                continue
            if node.op_type == CONVOLUTION:
                conv_params = self.get_conv_params(node)
                block_quantize_res = self.block_quantize(conv_params.weights)

                quantized_weights_name = f"{node.name}_quantized_weights"
                quantized_node_name = f"{node.name}_quantized_node"
                dequantized_weights_name = f"{node.name}_dequantized_weights"
                scales_name = f"{node.name}_scales"
                zero_point_name = f"{node.name}_zero_point"

                shape_node_name = f"{node.name}_shape_node"
                shape_name = f"{node.name}_shape"
                reshaped_weights_name = f"{node.name}_reshaped_weights"

                dequantize_node = create_dequantize_node(
                    quantized_node_name,
                    quantized_weights_name,
                    scales_name,
                    zero_point_name,
                    dequantized_weights_name,
                    block_quantize_res.block_size,
                    block_quantize_res.axis,
                )
                reshape_node = create_reshape_node(
                    shape_node_name,
                    dequantized_weights_name,
                    shape_name,
                    reshaped_weights_name,
                )

                shape_tensor = onnx.numpy_helper.from_array(
                    np.array(block_quantize_res.original_shape), name=shape_name
                )
                scale_initializer = onnx.numpy_helper.from_array(
                    block_quantize_res.scales, name=scales_name
                )
                zero_point_initializer = onnx.numpy_helper.from_array(
                    block_quantize_res.zero_point, name=zero_point_name
                )
                quantized_weights_initializer = onnx.numpy_helper.from_array(
                    block_quantize_res.quantized_weights, name=quantized_weights_name
                )

                dequantized_weights_info = helper.make_tensor_value_info(
                    dequantized_weights_name,
                    onnx.TensorProto.FLOAT,
                    block_quantize_res.quantized_weights.shape,
                )
                shape_info = helper.make_tensor_value_info(
                    reshaped_weights_name,
                    onnx.TensorProto.FLOAT,
                    block_quantize_res.original_shape,
                )

                self.graph.initializer.extend(
                    [
                        scale_initializer,
                        zero_point_initializer,
                        shape_tensor,
                        quantized_weights_initializer,
                    ]
                )

                # Removing fp32 weights
                self.graph.initializer.remove(
                    next(
                        init
                        for init in self.graph.initializer
                        if init.name == node.input[1]
                    )
                )
                node.input[1] = reshaped_weights_name

                # Preserving the topological order of graph nodes
                self.graph.node.insert(0, reshape_node)
                self.graph.node.insert(0, dequantize_node)
                self.graph.value_info.insert(0, shape_info)
                self.graph.value_info.insert(0, dequantized_weights_info)

                sqe.append(block_quantize_res.quantization_error**2)
                visited_nodes.append(node.name)

        onnx.checker.check_model(self.model, full_check=True)
        onnx.save(self.model, self.conf.output_model_path)

        self.display_summary(sqe)


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blockwise quantization tool")

    parser.add_argument(
        "-i",
        "--input_model",
        type=str,
        help="The path of onnx model to quantize",
        required=True,
    )
    parser.add_argument(
        "-bs",
        "--block_size",
        type=int,
        help="The maximum size of quantization block",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--bits",
        type=int,
        help="Quantization bits",
        choices=[8, 16],
        default=8,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output_model",
        type=str,
        help="The output model path",
        default="block_quantized_model.onnx",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = setup_args()

    quantization_config = BlockQuantizeConfig(
        input_model_path=args.input_model,
        output_model_path=args.output_model,
        block_size=args.block_size,
        bits=args.bits,
    )

    quantizer = BlockQuantizer(quantization_config)
    quantizer.run()
