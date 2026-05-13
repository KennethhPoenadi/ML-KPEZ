from __future__ import annotations

import numpy as np

from tubes2_ml.scratch.layers.conv import Conv2D, _compute_padding
from tubes2_ml.scratch.layers.dense import Dense
from tubes2_ml.scratch.layers.flatten import Flatten
from tubes2_ml.scratch.layers.locally_connected import LocallyConnected2D
from tubes2_ml.scratch.layers.pooling import (
    AveragePooling2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPooling2D,
)


class ScratchCNNClassifier:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.asarray(x, dtype=np.float32)
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=-1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = np.asarray(grad_output, dtype=np.float32)
        for layer in reversed(self.layers):
            if not hasattr(layer, "backward"):
                raise ValueError(f"Layer {layer.__class__.__name__} does not implement backward")
            grad = layer.backward(grad)
        return grad

    def count_parameters(self) -> int:
        total = 0
        for layer in self.layers:
            for attribute in ("kernel", "weights", "bias"):
                value = getattr(layer, attribute, None)
                if value is not None:
                    total += int(np.prod(value.shape))
        return total


def build_scratch_cnn_from_keras(
    keras_model,
    replace_conv_with_local: bool = False,
    input_shape: tuple[int, int, int] | None = None,
) -> ScratchCNNClassifier:
    layers = []
    current_shape = input_shape or tuple(int(dim) for dim in keras_model.input_shape[1:])

    for keras_layer in keras_model.layers:
        layer_type = keras_layer.__class__.__name__

        if layer_type in {"InputLayer", "Dropout"}:
            continue

        if layer_type == "Conv2D":
            scratch_layer, current_shape = _convert_conv2d(
                keras_layer,
                current_shape,
                replace_conv_with_local=replace_conv_with_local,
            )
            layers.append(scratch_layer)
            continue

        if layer_type == "MaxPooling2D":
            layers.append(
                MaxPooling2D(
                    pool_size=tuple(keras_layer.pool_size),
                    strides=tuple(keras_layer.strides),
                    padding=keras_layer.padding,
                )
            )
            current_shape = _pool_output_shape(current_shape, keras_layer.pool_size, keras_layer.strides, keras_layer.padding)
            continue

        if layer_type == "AveragePooling2D":
            layers.append(
                AveragePooling2D(
                    pool_size=tuple(keras_layer.pool_size),
                    strides=tuple(keras_layer.strides),
                    padding=keras_layer.padding,
                )
            )
            current_shape = _pool_output_shape(current_shape, keras_layer.pool_size, keras_layer.strides, keras_layer.padding)
            continue

        if layer_type == "GlobalAveragePooling2D":
            layers.append(GlobalAveragePooling2D())
            current_shape = (current_shape[-1],)
            continue

        if layer_type == "GlobalMaxPooling2D":
            layers.append(GlobalMaxPooling2D())
            current_shape = (current_shape[-1],)
            continue

        if layer_type == "Flatten":
            layers.append(Flatten())
            current_shape = (int(np.prod(current_shape)),)
            continue

        if layer_type == "Dense":
            weights = keras_layer.get_weights()
            dense = Dense(activation=_activation_name(keras_layer.activation))
            dense.load_keras_weights(weights)
            layers.append(dense)
            current_shape = (weights[0].shape[-1],)
            continue

        raise ValueError(f"Unsupported Keras layer for scratch CNN: {layer_type}")

    return ScratchCNNClassifier(layers)


def _convert_conv2d(keras_layer, input_shape, replace_conv_with_local: bool):
    kernel, bias = keras_layer.get_weights()
    kernel_size = tuple(keras_layer.kernel_size)
    strides = tuple(keras_layer.strides)
    padding = keras_layer.padding
    activation = _activation_name(keras_layer.activation)
    output_shape = _conv_output_shape(input_shape, kernel_size, strides, padding, kernel.shape[-1])

    if not replace_conv_with_local:
        layer = Conv2D(strides=strides, padding=padding, activation=activation)
        layer.load_keras_weights([kernel, bias])
        return layer, output_shape

    output_positions = output_shape[0] * output_shape[1]
    local_kernel = kernel.reshape(-1, kernel.shape[-1])
    local_kernel = np.repeat(local_kernel[np.newaxis, :, :], output_positions, axis=0)
    local_bias = np.repeat(bias[np.newaxis, :], output_positions, axis=0)

    layer = LocallyConnected2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
    )
    layer.load_keras_weights([local_kernel, local_bias])
    return layer, output_shape


def _activation_name(activation) -> str | None:
    name = getattr(activation, "__name__", None)
    return None if name in {None, "linear"} else name


def _conv_output_shape(input_shape, kernel_size, strides, padding, filters):
    height, width, _ = input_shape
    _, _, output_height = _compute_padding(height, kernel_size[0], strides[0], padding)
    _, _, output_width = _compute_padding(width, kernel_size[1], strides[1], padding)
    return output_height, output_width, filters


def _pool_output_shape(input_shape, pool_size, strides, padding):
    height, width, channels = input_shape
    _, _, output_height = _compute_padding(height, pool_size[0], strides[0], padding)
    _, _, output_width = _compute_padding(width, pool_size[1], strides[1], padding)
    return output_height, output_width, channels
