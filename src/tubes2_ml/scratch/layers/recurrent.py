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
        prev_states: list[np.ndarray] = []
        states: list[np.ndarray] = []
        bias = 0.0 if self.bias is None else self.bias
        for timestep in range(timesteps):
            prev_states.append(h_t)
            h_t = tanh(inputs[:, timestep, :] @ self.kernel + h_t @ self.recurrent_kernel + bias)
            states.append(h_t)
            outputs.append(h_t)

        sequence = np.stack(outputs, axis=1) if outputs else np.empty((batch_size, 0, self.units), dtype=np.float32)
        self._cache = {
            "inputs": inputs,
            "prev_states": prev_states,
            "states": states,
            "initial_state": np.zeros((batch_size, self.units), dtype=np.float32)
            if initial_state is None
            else np.asarray(initial_state, dtype=np.float32),
        }
        if return_state:
            return sequence, h_t
        return sequence

    def backward(
        self,
        grad_outputs: np.ndarray,
        grad_final_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.kernel is None or self.recurrent_kernel is None:
            raise ValueError("SimpleRNN weights are not loaded")
        if not hasattr(self, "_cache"):
            raise ValueError("SimpleRNN backward called before forward")

        inputs = self._cache["inputs"]
        prev_states = self._cache["prev_states"]
        states = self._cache["states"]
        grad_sequence = np.asarray(grad_outputs, dtype=np.float32)
        batch_size, timesteps, input_dim = inputs.shape

        self.grad_kernel = np.zeros_like(self.kernel)
        self.grad_recurrent_kernel = np.zeros_like(self.recurrent_kernel)
        self.grad_bias = np.zeros_like(self.bias) if self.bias is not None else None
        grad_inputs = np.zeros_like(inputs)
        grad_h_next = (
            np.zeros((batch_size, self.units), dtype=np.float32)
            if grad_final_state is None
            else np.asarray(grad_final_state, dtype=np.float32)
        )

        for timestep in range(timesteps - 1, -1, -1):
            grad_h = grad_sequence[:, timestep, :] + grad_h_next
            h_t = states[timestep]
            prev_h = prev_states[timestep]
            grad_z = grad_h * (1.0 - h_t * h_t)
            self.grad_kernel += inputs[:, timestep, :].T @ grad_z
            self.grad_recurrent_kernel += prev_h.T @ grad_z
            if self.grad_bias is not None:
                self.grad_bias += np.sum(grad_z, axis=0)
            grad_inputs[:, timestep, :] = grad_z @ self.kernel.T
            grad_h_next = grad_z @ self.recurrent_kernel.T

        return grad_inputs.reshape(batch_size, timesteps, input_dim), grad_h_next


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
        caches: list[dict[str, np.ndarray]] = []
        bias = 0.0 if self.bias is None else self.bias
        for timestep in range(timesteps):
            prev_h = h_t
            prev_c = c_t
            z = inputs[:, timestep, :] @ self.kernel + h_t @ self.recurrent_kernel + bias
            z_i, z_f, z_c, z_o = np.split(z, 4, axis=-1)
            i_t = sigmoid(z_i)
            f_t = sigmoid(z_f)
            c_hat_t = tanh(z_c)
            o_t = sigmoid(z_o)
            c_t = f_t * c_t + i_t * c_hat_t
            h_t = o_t * tanh(c_t)
            caches.append(
                {
                    "prev_h": prev_h,
                    "prev_c": prev_c,
                    "i": i_t,
                    "f": f_t,
                    "c_hat": c_hat_t,
                    "o": o_t,
                    "c": c_t,
                }
            )
            outputs.append(h_t)

        sequence = np.stack(outputs, axis=1) if outputs else np.empty((batch_size, 0, self.units), dtype=np.float32)
        self._cache = {
            "inputs": inputs,
            "step_caches": caches,
            "initial_state": (
                np.zeros((batch_size, self.units), dtype=np.float32),
                np.zeros((batch_size, self.units), dtype=np.float32),
            )
            if initial_state is None
            else (
                np.asarray(initial_state[0], dtype=np.float32),
                np.asarray(initial_state[1], dtype=np.float32),
            ),
        }
        if return_state:
            return sequence, h_t, c_t
        return sequence

    def backward(
        self,
        grad_outputs: np.ndarray,
        grad_final_state: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.kernel is None or self.recurrent_kernel is None:
            raise ValueError("LSTM weights are not loaded")
        if not hasattr(self, "_cache"):
            raise ValueError("LSTM backward called before forward")

        inputs = self._cache["inputs"]
        step_caches = self._cache["step_caches"]
        grad_sequence = np.asarray(grad_outputs, dtype=np.float32)
        batch_size, timesteps, input_dim = inputs.shape

        self.grad_kernel = np.zeros_like(self.kernel)
        self.grad_recurrent_kernel = np.zeros_like(self.recurrent_kernel)
        self.grad_bias = np.zeros_like(self.bias) if self.bias is not None else None
        grad_inputs = np.zeros_like(inputs)
        if grad_final_state is None:
            grad_h_next = np.zeros((batch_size, self.units), dtype=np.float32)
            grad_c_next = np.zeros((batch_size, self.units), dtype=np.float32)
        else:
            grad_h_next = np.asarray(grad_final_state[0], dtype=np.float32)
            grad_c_next = np.asarray(grad_final_state[1], dtype=np.float32)

        for timestep in range(timesteps - 1, -1, -1):
            cache = step_caches[timestep]
            grad_h = grad_sequence[:, timestep, :] + grad_h_next
            tanh_c = tanh(cache["c"])
            grad_o = grad_h * tanh_c
            grad_c = grad_h * cache["o"] * (1.0 - tanh_c * tanh_c) + grad_c_next
            grad_f = grad_c * cache["prev_c"]
            grad_prev_c = grad_c * cache["f"]
            grad_i = grad_c * cache["c_hat"]
            grad_c_hat = grad_c * cache["i"]

            grad_z_i = grad_i * cache["i"] * (1.0 - cache["i"])
            grad_z_f = grad_f * cache["f"] * (1.0 - cache["f"])
            grad_z_c = grad_c_hat * (1.0 - cache["c_hat"] * cache["c_hat"])
            grad_z_o = grad_o * cache["o"] * (1.0 - cache["o"])
            grad_z = np.concatenate([grad_z_i, grad_z_f, grad_z_c, grad_z_o], axis=-1)

            self.grad_kernel += inputs[:, timestep, :].T @ grad_z
            self.grad_recurrent_kernel += cache["prev_h"].T @ grad_z
            if self.grad_bias is not None:
                self.grad_bias += np.sum(grad_z, axis=0)
            grad_inputs[:, timestep, :] = grad_z @ self.kernel.T
            grad_h_next = grad_z @ self.recurrent_kernel.T
            grad_c_next = grad_prev_c

        return grad_inputs.reshape(batch_size, timesteps, input_dim), grad_h_next, grad_c_next
