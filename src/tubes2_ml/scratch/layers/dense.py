from __future__ import annotations
import numpy as np
from tubes2_ml.scratch.layers.activations import activation_backward, get_activation

class Dense:
    def __init__(self,weights: np.ndarray | None = None,bias: np.ndarray | None = None,activation: str | None = None):
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.activation_name = activation
        self.activation = get_activation(activation)

    def load_keras_weights(self, weights: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
        if len(weights) == 1:
            self.weights = np.asarray(weights[0], dtype=np.float32)
            self.bias = None
        elif len(weights) == 2:
            self.weights = np.asarray(weights[0], dtype=np.float32)
            self.bias = np.asarray(weights[1], dtype=np.float32)
        else:
            raise ValueError("Dense weights must be [kernel] or [kernel, bias]")

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Dense weights are not loaded")

        x = np.asarray(x, dtype=np.float32)
        output = x @ self.weights
        if self.bias is not None:
            output += self.bias

        self._input = x
        self._pre_activation = output
        return self.activation(output)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Dense weights are not loaded")
        if not hasattr(self, "_input") or not hasattr(self, "_pre_activation"):
            raise ValueError("Dense backward called before forward")

        grad = activation_backward(
            self.activation_name,
            self._pre_activation,
            np.asarray(grad_output, dtype=np.float32),
        )
        input_shape = self._input.shape
        flat_input = self._input.reshape(-1, self.weights.shape[0])
        flat_grad = grad.reshape(-1, self.weights.shape[1])

        self.grad_weights = flat_input.T @ flat_grad
        self.grad_bias = np.sum(flat_grad, axis=0) if self.bias is not None else None
        grad_input = flat_grad @ self.weights.T
        return grad_input.reshape(input_shape)
