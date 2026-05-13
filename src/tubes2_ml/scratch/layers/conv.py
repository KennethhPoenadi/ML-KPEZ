from __future__ import annotations
import numpy as np
from tubes2_ml.scratch.layers.activations import activation_backward, get_activation

def _to_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("Expected a pair of integers")
        return value
    return value, value

def _compute_padding(input_size: int,kernel_size: int,stride: int,padding: str) -> tuple[int, int, int]:
    if padding == "valid":
        output_size = (input_size - kernel_size) // stride + 1
        return 0, 0, output_size
    if padding == "same":
        output_size = int(np.ceil(input_size / stride))
        total_padding = max((output_size - 1) * stride + kernel_size - input_size, 0)
        before = total_padding // 2
        after = total_padding - before
        return before, after, output_size
    
    raise ValueError("padding must be either 'valid' or 'same'")


class Conv2D:
    def __init__(self,kernel: np.ndarray | None = None,bias: np.ndarray | None = None,strides: int | tuple[int, int] = 1,padding: str = "valid",activation: str | None = None):
        self.kernel = None if kernel is None else np.asarray(kernel, dtype=np.float32)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
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
            raise ValueError("Conv2D weights must be [kernel] or [kernel, bias]")

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            raise ValueError("Conv2D kernel is not loaded")

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError("Conv2D input must have shape (N, H, W, C)")

        batch_size, input_height, input_width, input_channels = x.shape
        kernel_height, kernel_width, kernel_channels, output_channels = self.kernel.shape
        if input_channels != kernel_channels:
            raise ValueError("Input channels do not match Conv2D kernel channels")

        stride_h, stride_w = self.strides
        pad_top, pad_bottom, output_height = _compute_padding(
            input_height, kernel_height, stride_h, self.padding
        )
        pad_left, pad_right, output_width = _compute_padding(
            input_width, kernel_width, stride_w, self.padding
        )

        padded = np.pad(x,((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),mode="constant")
        output = np.zeros((batch_size, output_height, output_width, output_channels), dtype=np.float32)

        for row in range(output_height):
            row_start = row * stride_h
            row_end = row_start + kernel_height
            for col in range(output_width):
                col_start = col * stride_w
                col_end = col_start + kernel_width
                patch = padded[:, row_start:row_end, col_start:col_end, :]
                output[:, row, col, :] = np.tensordot(
                    patch,
                    self.kernel,
                    axes=((1, 2, 3), (0, 1, 2)),
                )

        if self.bias is not None:
            output += self.bias.reshape(1, 1, 1, output_channels)

        self._cache = {
            "input_shape": x.shape,
            "padded": padded,
            "padding": (pad_top, pad_bottom, pad_left, pad_right),
            "pre_activation": output,
        }
        return self.activation(output)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            raise ValueError("Conv2D kernel is not loaded")
        if not hasattr(self, "_cache"):
            raise ValueError("Conv2D backward called before forward")

        grad = activation_backward(
            self.activation_name,
            self._cache["pre_activation"],
            np.asarray(grad_output, dtype=np.float32),
        )
        padded = self._cache["padded"]
        pad_top, pad_bottom, pad_left, pad_right = self._cache["padding"]
        kernel_height, kernel_width, _, _ = self.kernel.shape
        _, output_height, output_width, _ = grad.shape
        stride_h, stride_w = self.strides

        self.grad_kernel = np.zeros_like(self.kernel)
        self.grad_bias = np.sum(grad, axis=(0, 1, 2)) if self.bias is not None else None
        grad_padded = np.zeros_like(padded)

        for row in range(output_height):
            row_start = row * stride_h
            row_end = row_start + kernel_height
            for col in range(output_width):
                col_start = col * stride_w
                col_end = col_start + kernel_width
                patch = padded[:, row_start:row_end, col_start:col_end, :]
                grad_position = grad[:, row, col, :]
                self.grad_kernel += np.tensordot(patch, grad_position, axes=([0], [0]))
                grad_padded[:, row_start:row_end, col_start:col_end, :] += np.tensordot(
                    grad_position,
                    self.kernel,
                    axes=([1], [3]),
                )

        row_slice = slice(pad_top, grad_padded.shape[1] - pad_bottom if pad_bottom else None)
        col_slice = slice(pad_left, grad_padded.shape[2] - pad_right if pad_right else None)
        return grad_padded[:, row_slice, col_slice, :].reshape(self._cache["input_shape"])
