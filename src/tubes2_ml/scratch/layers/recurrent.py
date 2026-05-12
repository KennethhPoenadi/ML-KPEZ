from __future__ import annotations

import numpy as np

from tubes2_ml.scratch.layers.activations import sigmoid, tanh


class SimpleRNN:
    def __init__(
        self,
        kernel: np.ndarray | None = None,
        recurrent_kernel: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ):
        self.kernel = None if kernel is None else np.asarray(kernel, dtype=np.float32)
        self.recurrent_kernel = (
            None if recurrent_kernel is None else np.asarray(recurrent_kernel, dtype=np.float32)
        )
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)

    def load_keras_weights(self, weights: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
        if len(weights) != 3:
            raise ValueError("SimpleRNN weights must be [kernel, recurrent_kernel, bias]")
        self.kernel = np.asarray(weights[0], dtype=np.float32)
        self.recurrent_kernel = np.asarray(weights[1], dtype=np.float32)
        self.bias = np.asarray(weights[2], dtype=np.float32)

    @property
    def units(self) -> int:
        if self.recurrent_kernel is None:
            raise ValueError("SimpleRNN weights are not loaded")
        return int(self.recurrent_kernel.shape[0])

    def forward(
        self,
        x: np.ndarray,
        initial_state: np.ndarray | None = None,
        return_state: bool = False,
    ):
        if self.kernel is None or self.recurrent_kernel is None:
            raise ValueError("SimpleRNN weights are not loaded")

        inputs = np.asarray(x, dtype=np.float32)
        if inputs.ndim != 3:
            raise ValueError("SimpleRNN input must have shape (batch, timesteps, features)")

        batch_size, timesteps, _ = inputs.shape
        if initial_state is None:
            h_t = np.zeros((batch_size, self.units), dtype=np.float32)
        else:
            h_t = np.asarray(initial_state, dtype=np.float32)

        outputs: list[np.ndarray] = []
        bias = 0.0 if self.bias is None else self.bias
        for timestep in range(timesteps):
            h_t = tanh(inputs[:, timestep, :] @ self.kernel + h_t @ self.recurrent_kernel + bias)
            outputs.append(h_t)

        sequence = np.stack(outputs, axis=1) if outputs else np.empty((batch_size, 0, self.units), dtype=np.float32)
        if return_state:
            return sequence, h_t
        return sequence


class LSTM:
    def __init__(
        self,
        kernel: np.ndarray | None = None,
        recurrent_kernel: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ):
        self.kernel = None if kernel is None else np.asarray(kernel, dtype=np.float32)
        self.recurrent_kernel = (
            None if recurrent_kernel is None else np.asarray(recurrent_kernel, dtype=np.float32)
        )
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)

    def load_keras_weights(self, weights: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
        if len(weights) != 3:
            raise ValueError("LSTM weights must be [kernel, recurrent_kernel, bias]")
        self.kernel = np.asarray(weights[0], dtype=np.float32)
        self.recurrent_kernel = np.asarray(weights[1], dtype=np.float32)
        self.bias = np.asarray(weights[2], dtype=np.float32)

    @property
    def units(self) -> int:
        if self.recurrent_kernel is None:
            raise ValueError("LSTM weights are not loaded")
        return int(self.recurrent_kernel.shape[0])

    def forward(
        self,
        x: np.ndarray,
        initial_state: tuple[np.ndarray, np.ndarray] | None = None,
        return_state: bool = False,
    ):
        if self.kernel is None or self.recurrent_kernel is None:
            raise ValueError("LSTM weights are not loaded")

        inputs = np.asarray(x, dtype=np.float32)
        if inputs.ndim != 3:
            raise ValueError("LSTM input must have shape (batch, timesteps, features)")

        batch_size, timesteps, _ = inputs.shape
        if initial_state is None:
            h_t = np.zeros((batch_size, self.units), dtype=np.float32)
            c_t = np.zeros((batch_size, self.units), dtype=np.float32)
        else:
            h_t = np.asarray(initial_state[0], dtype=np.float32)
            c_t = np.asarray(initial_state[1], dtype=np.float32)

        outputs: list[np.ndarray] = []
        bias = 0.0 if self.bias is None else self.bias
        for timestep in range(timesteps):
            z = inputs[:, timestep, :] @ self.kernel + h_t @ self.recurrent_kernel + bias
            z_i, z_f, z_c, z_o = np.split(z, 4, axis=-1)
            i_t = sigmoid(z_i)
            f_t = sigmoid(z_f)
            c_hat_t = tanh(z_c)
            o_t = sigmoid(z_o)
            c_t = f_t * c_t + i_t * c_hat_t
            h_t = o_t * tanh(c_t)
            outputs.append(h_t)

        sequence = np.stack(outputs, axis=1) if outputs else np.empty((batch_size, 0, self.units), dtype=np.float32)
        if return_state:
            return sequence, h_t, c_t
        return sequence
