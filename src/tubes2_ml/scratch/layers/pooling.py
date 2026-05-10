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
        return output

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
        return np.mean(x, axis=(1, 2))

class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError("GlobalMaxPooling2D input must have shape (N, H, W, C)")
        return np.max(x, axis=(1, 2))