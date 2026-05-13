from __future__ import annotations
import numpy as np
from tubes2_ml.scratch.layers.conv import _compute_padding, _to_pair

class Pooling2D:
    def __init__(self,pool_size: int | tuple[int, int] = (2, 2),strides: int | tuple[int, int] | None = None,padding: str = "valid",mode: str = "max"):

        if mode not in {"max", "average"}:
            raise ValueError("mode must be either 'max' or 'average'")
        
        self.pool_size = _to_pair(pool_size)
        self.strides = _to_pair(strides if strides is not None else pool_size)
        self.padding = padding.lower()
        self.mode = mode

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError("Pooling2D input must have shape (N, H, W, C)")

        batch_size, input_height, input_width, channels = x.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.strides
        pad_top, pad_bottom, output_height = _compute_padding(
            input_height, pool_h, stride_h, self.padding
        )
        pad_left, pad_right, output_width = _compute_padding(
            input_width, pool_w, stride_w, self.padding
        )

        pad_value = -np.inf if self.mode == "max" else 0.0
        padded = np.pad(x, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=pad_value)
        output = np.zeros((batch_size, output_height, output_width, channels), dtype=np.float32)

        for row in range(output_height):
            row_start = row * stride_h
            row_end = row_start + pool_h
            for col in range(output_width):
                col_start = col * stride_w
                col_end = col_start + pool_w
                patch = padded[:, row_start:row_end, col_start:col_end, :]
                if self.mode == "max":
                    output[:, row, col, :] = np.max(patch, axis=(1, 2))
                else:
                    output[:, row, col, :] = np.mean(patch, axis=(1, 2))
        self._cache = {
            "input_shape": x.shape,
            "padded": padded,
            "padding": (pad_top, pad_bottom, pad_left, pad_right),
            "output": output,
        }
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_cache"):
            raise ValueError("Pooling2D backward called before forward")

        grad = np.asarray(grad_output, dtype=np.float32)
        padded = self._cache["padded"]
        pad_top, pad_bottom, pad_left, pad_right = self._cache["padding"]
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.strides
        _, output_height, output_width, _ = grad.shape
        grad_padded = np.zeros_like(padded)

        for row in range(output_height):
            row_start = row * stride_h
            row_end = row_start + pool_h
            for col in range(output_width):
                col_start = col * stride_w
                col_end = col_start + pool_w
                patch = padded[:, row_start:row_end, col_start:col_end, :]
                grad_position = grad[:, row, col, :][:, np.newaxis, np.newaxis, :]
                if self.mode == "max":
                    maxima = np.max(patch, axis=(1, 2), keepdims=True)
                    mask = patch == maxima
                    counts = np.sum(mask, axis=(1, 2), keepdims=True)
                    grad_padded[:, row_start:row_end, col_start:col_end, :] += mask * grad_position / counts
                else:
                    grad_padded[:, row_start:row_end, col_start:col_end, :] += grad_position / (pool_h * pool_w)

        row_slice = slice(pad_top, grad_padded.shape[1] - pad_bottom if pad_bottom else None)
        col_slice = slice(pad_left, grad_padded.shape[2] - pad_right if pad_right else None)
        return grad_padded[:, row_slice, col_slice, :].reshape(self._cache["input_shape"])

class MaxPooling2D(Pooling2D):
    def __init__(self,pool_size: int | tuple[int, int] = (2, 2),strides: int | tuple[int, int] | None = None,padding: str = "valid"):
        super().__init__(pool_size=pool_size, strides=strides, padding=padding, mode="max")

class AveragePooling2D(Pooling2D):
    def __init__(self,pool_size: int | tuple[int, int] = (2, 2), strides: int | tuple[int, int] | None = None, padding: str = "valid"):
        super().__init__(pool_size=pool_size, strides=strides, padding=padding, mode="average")

class GlobalAveragePooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError("GlobalAveragePooling2D input must have shape (N, H, W, C)")
        self._input_shape = x.shape
        return np.mean(x, axis=(1, 2))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_input_shape"):
            raise ValueError("GlobalAveragePooling2D backward called before forward")
        batch_size, height, width, channels = self._input_shape
        grad = np.asarray(grad_output, dtype=np.float32).reshape(batch_size, 1, 1, channels)
        return np.ones(self._input_shape, dtype=np.float32) * grad / (height * width)

class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError("GlobalMaxPooling2D input must have shape (N, H, W, C)")
        self._input = x
        output = np.max(x, axis=(1, 2))
        self._output = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_input") or not hasattr(self, "_output"):
            raise ValueError("GlobalMaxPooling2D backward called before forward")
        grad = np.asarray(grad_output, dtype=np.float32)[:, np.newaxis, np.newaxis, :]
        maxima = self._output[:, np.newaxis, np.newaxis, :]
        mask = self._input == maxima
        counts = np.sum(mask, axis=(1, 2), keepdims=True)
        return mask * grad / counts
