from __future__ import annotations
import numpy as np
from tubes2_ml.scratch.layers.activations import get_activation

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
            
        return self.activation(output)
