from __future__ import annotations
import numpy as np
from tubes2_ml.scratch.layers.activations import get_activation
from tubes2_ml.scratch.layers.conv import _compute_padding, _to_pair

class LocallyConnected2D:
    def __init__(self,kernel: np.ndarray | None = None,bias: np.ndarray | None = None,kernel_size: int | tuple[int, int] | None = None,filters: int | None = None,strides: int | tuple[int, int] = 1,padding: str = "valid",activation: str | None = None):
        
        self.kernel = None if kernel is None else np.asarray(kernel, dtype=np.float32)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.kernel_size = None if kernel_size is None else _to_pair(kernel_size)
        self.filters = filters
        self.strides = _to_pair(strides)
        self.padding = padding.lower()
        self.activation_name = activation
        self.activation = get_activation(activation)

    def load_keras_weights(self, weights: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:

        if len(weights) == 1:
            self.kernel = np.asarray(weights[0], dtype=np.float32)
            self.bias = None

        elif len(weights) == 2:
            self.kernel = np.asarray(weights[0], dtype=np.float32)
            self.bias = np.asarray(weights[1], dtype=np.float32)

        else:
            raise ValueError("LocallyConnected2D weights must be [kernel] or [kernel, bias]")

        if self.kernel.ndim != 3:
            raise ValueError("LocallyConnected2D kernel must have shape (out_positions, patch_size, filters)")
        
        self.filters = int(self.kernel.shape[-1])

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.kernel is None:
            raise ValueError("LocallyConnected2D kernel is not loaded")
        if self.kernel_size is None:
            raise ValueError("kernel_size is required for LocallyConnected2D forward pass")

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError("LocallyConnected2D input must have shape (N, H, W, C)")

        batch_size, input_height, input_width, input_channels = x.shape
        kernel_height, kernel_width = self.kernel_size
        stride_h, stride_w = self.strides
        pad_top, pad_bottom, output_height = _compute_padding(
            input_height, kernel_height, stride_h, self.padding
        )
        pad_left, pad_right, output_width = _compute_padding(
            input_width, kernel_width, stride_w, self.padding
        )

        expected_positions = output_height * output_width
        patch_size = kernel_height * kernel_width * input_channels
        if self.kernel.shape[0] != expected_positions:
            raise ValueError("Kernel output positions do not match computed output size")
        if self.kernel.shape[1] != patch_size:
            raise ValueError("Kernel patch size does not match input channels and kernel_size")

        output_channels = self.kernel.shape[2]
        padded = np.pad(
            x,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )
        output = np.zeros((batch_size, output_height, output_width, output_channels), dtype=np.float32)

        position = 0
        for row in range(output_height):
            row_start = row * stride_h
            row_end = row_start + kernel_height
            for col in range(output_width):
                col_start = col * stride_w
                col_end = col_start + kernel_width
                patch = padded[:, row_start:row_end, col_start:col_end, :].reshape(batch_size, -1)
                output[:, row, col, :] = patch @ self.kernel[position]
                position += 1

        if self.bias is not None:
            output += self._reshape_bias(self.bias, output_height, output_width, output_channels)

        return self.activation(output)

    @staticmethod
    def _reshape_bias(bias: np.ndarray,output_height: int,output_width: int,output_channels: int) -> np.ndarray:

        if bias.shape == (output_height, output_width, output_channels):
            return bias.reshape(1, output_height, output_width, output_channels)
        if bias.shape == (output_height * output_width, output_channels):
            return bias.reshape(1, output_height, output_width, output_channels)
        if bias.shape == (output_channels,):
            return bias.reshape(1, 1, 1, output_channels)
        raise ValueError("Unsupported LocallyConnected2D bias shape")
