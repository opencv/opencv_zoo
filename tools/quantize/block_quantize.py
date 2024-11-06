import sys

MIN_PYTHON_VERSION = (3, 7)

if sys.version_info < MIN_PYTHON_VERSION:
    raise ImportError("This script requires Python 3.7 or higher!")

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple
from enum import Enum, auto

import numpy as np
import onnx
from onnx import helper

BITS_TO_NUMPY_TYPE = {8: np.int8, 16: np.int16}


SUPPORTED_OPS = {"Conv", "Gemm", "MatMul"}

ONNX_OPSET = 21


class WeightCategory(Enum):
    INITIALIZER = auto()
    CONSTANT = auto()
    NONE = auto()


@dataclass
class BlockQuantizeConfig:
    input_model_path: str
    output_model_path: str
    block_size: int
    bits: int
    verbose: bool


@dataclass
class BlockQuantizeResult:
    quantized_weights: np.ndarray = field(default_factory=lambda: np.array([]))
    scales: np.ndarray = field(default_factory=lambda: np.array([]))
    zero_point: np.ndarray = field(default_factory=lambda: np.array([]))
    block_size: int = 1
    axis: int = 1
    original_shape: Tuple = field(default_factory=tuple)
    quantization_error: np.ndarray = field(default_factory=lambda: np.array([]))


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

    type_info = np.iinfo(BITS_TO_NUMPY_TYPE[n_bits])
    min_value = type_info.min
    max_value = type_info.max

    y = np.rint(x / y_scale_elementwise + y_zero_point_elementwise)
    y = np.clip(y, min_value, max_value)
    y = y.astype(BITS_TO_NUMPY_TYPE[n_bits])

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
        self.costants_map = {
            node.output[0]: next(
                attr.t for attr in node.attribute if attr.name == "value"
            )
            for node in self.model.graph.node
            if node.op_type == "Constant"
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

    def get_weight_category(self, name: str) -> WeightCategory:
        if name in self.initializers_map:
            return WeightCategory.INITIALIZER
        if name in self.costants_map:
            return WeightCategory.CONSTANT
        else:
            return WeightCategory.NONE

    def get_weight_tensor(self, name: str, category: WeightCategory) -> np.ndarray:
        if category == WeightCategory.INITIALIZER:
            return onnx.numpy_helper.to_array(self.initializers_map[name])
        elif category == WeightCategory.CONSTANT:
            return onnx.numpy_helper.to_array(self.costants_map[name])
        else:
            raise AssertionError("Invalid weight category")

    def remove_fp32_weights(self, name: str, category: WeightCategory):
        if category == WeightCategory.INITIALIZER:
            self.graph.initializer.remove(
                next(init for init in self.graph.initializer if init.name == name)
            )
        elif category == WeightCategory.CONSTANT:
            self.graph.node.remove(
                next(
                    node
                    for node in self.graph.node
                    if node.op_type == "Constant" and node.output[0] == name
                )
            )
        else:
            raise AssertionError("Invalid weight category")

    def compute_scale_zeropoint(
        self, b_min: np.ndarray, b_max: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert (
            b_min <= b_max
        ).all(), "minimum must not be greater than maximum when computing scale and zero point"

        # zero must be present in the range, this enforces qmin <= zero_point <= qmax
        b_min = np.minimum(b_min, np.zeros_like(b_min, dtype=b_min.dtype))
        b_max = np.maximum(b_max, np.zeros_like(b_max, dtype=b_max.dtype))

        type_info = np.iinfo(BITS_TO_NUMPY_TYPE[self.conf.bits])
        qmin = type_info.min
        qmax = type_info.max

        dq = qmax - qmin

        scales = np.where(b_max != b_min, (b_max - b_min) / dq, 1.0)

        zeropoints = np.where(b_max != b_min, np.rint(qmin - b_min / scales), 0.0)
        zeropoints = zeropoints.astype(BITS_TO_NUMPY_TYPE[self.conf.bits])

        return (scales, zeropoints)

    def block_quantize(self, weight: np.ndarray) -> BlockQuantizeResult:
        original_shape = weight.shape

        if weight.ndim > 1:
            weight = weight.reshape((weight.shape[0], -1))
            quantization_axis = 1
        else:
            quantization_axis = 0

        block_size = closest_divisor(
            weight.shape[quantization_axis], self.conf.block_size
        )

        assert (
            weight.shape[quantization_axis] % block_size == 0
        ), f"weight shape ({weight.shape[quantization_axis]}) must be divisible by block size ({block_size})"

        # Flattening the tensor after the quantization axis
        new_shape = list(weight.shape[: quantization_axis + 1]) + [-1]
        new_shape[quantization_axis] = new_shape[quantization_axis] // block_size

        blocked_weight = weight.reshape(new_shape)

        blocked_max = np.max(blocked_weight, -1)
        blocked_min = np.min(blocked_weight, -1)

        scales, zeropoints = self.compute_scale_zeropoint(blocked_min, blocked_max)

        quantized_weight = block_quantize_tensor(
            weight, quantization_axis, scales, zeropoints, self.conf.bits
        )
        reconstructed_mat = block_dequantize_tensor(
            quantized_weight, quantization_axis, scales, zeropoints
        )

        # Relative Norm
        qerror = np.linalg.norm(reconstructed_mat - weight) / (np.linalg.norm(weight) + 1e-10)

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

    def display_summary(self, sqe: Dict[str, int]):
        sqe_v = list(sqe.values())
        if len(sqe_v) == 0:
            mse = 0
            print(
                "Warning: No weights have been quantized, likely due to unsupported layers."
            )
        else:
            mse = sum(sqe_v) / len(sqe_v)
        original_model_size = self.get_model_size(self.conf.input_model_path)
        quantized_model_size = self.get_model_size(self.conf.output_model_path)

        if self.conf.verbose:
            sorted_sqe = sorted(sqe.items(), key=lambda item: item[1], reverse=True)
            longest_key_len = max(len(key) for key in sqe.keys())
            
            print("Quantization error (Relative Norm) sorted in ascending order:")

            for key, value in sorted_sqe:
                print(f"{key:<{longest_key_len}} : {value}")

        print("Done! Results saved in", self.conf.output_model_path)
        print("\nSummary of Results:\n")
        print(f"{'Metric':<30} {'Value':<10}")
        print(f"{'-'*40}")
        print(f"{'Relative Norm Error':<31} {mse:.6f}")
        print(f"{'Original Model Size (KB)':<31} {original_model_size:,.2f}")
        print(f"{'Block-Quantized Model Size (KB)':<30} {quantized_model_size:,.2f}")

    def run(self):
        print("Quantizing the model...")

        quantized_inputs = []
        sqe = {}

        node_idx = 0

        while node_idx < len(self.model.graph.node):
            node = self.model.graph.node[node_idx]

            if node.op_type in SUPPORTED_OPS:
                for input_idx, input_name in enumerate(node.input):
                    weightCategory = self.get_weight_category(input_name)

                    # Skip quantization if weights are taken as external input
                    if weightCategory == WeightCategory.NONE:
                        continue

                    weight = self.get_weight_tensor(input_name, weightCategory)

                    quantized_weights_name = f"{input_name}_quantized"
                    quantized_node_name = f"{input_name}_quantized_node"
                    dequantized_weights_name = f"{input_name}_dequantized"
                    scales_name = f"{input_name}_scales"
                    zero_point_name = f"{input_name}_zero_point"

                    shape_node_name = f"{input_name}_shape_node"
                    shape_name = f"{input_name}_shape"
                    reshaped_weights_name = f"{input_name}_reshaped"

                    # Skip quantization if weights don't contain enough elements to create at least 1 block
                    if weight.size < self.conf.block_size:
                        continue

                    reshape_needed = weight.ndim > 2

                    # In case of parameter sharing
                    if input_name in quantized_inputs:
                        node.input[input_idx] = (
                            reshaped_weights_name
                            if reshape_needed
                            else dequantized_weights_name
                        )
                        continue


                    block_quantize_res = self.block_quantize(weight)

                    # Skip quantization if it wouldn't reduce the model size
                    if block_quantize_res.block_size == 1:
                        continue

                    quantized_inputs.append(input_name)

                    dequantize_node = create_dequantize_node(
                        quantized_node_name,
                        quantized_weights_name,
                        scales_name,
                        zero_point_name,
                        dequantized_weights_name,
                        block_quantize_res.block_size,
                        block_quantize_res.axis,
                    )

                    if reshape_needed:
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
                        block_quantize_res.quantized_weights,
                        name=quantized_weights_name,
                    )

                    dequantized_weights_info = helper.make_tensor_value_info(
                        dequantized_weights_name,
                        onnx.TensorProto.FLOAT,
                        block_quantize_res.quantized_weights.shape,
                    )

                    if reshape_needed:
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

                    self.remove_fp32_weights(input_name, weightCategory)

                    node.input[input_idx] = (
                        reshaped_weights_name
                        if reshape_needed
                        else dequantized_weights_name
                    )

                    # Preserving graph nodes topological order
                    if reshape_needed:
                        self.graph.node.insert(0, reshape_node)
                        node_idx += 1

                    self.graph.node.insert(0, dequantize_node)
                    node_idx += 1
                    if reshape_needed:
                        self.graph.value_info.insert(0, shape_info)
                    self.graph.value_info.insert(0, dequantized_weights_info)

                    sqe[input_name] = block_quantize_res.quantization_error

            node_idx += 1

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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
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
        verbose=args.verbose
    )

    quantizer = BlockQuantizer(quantization_config)
    quantizer.run()
