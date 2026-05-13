from __future__ import annotations
import numpy as np

def linear(x: np.ndarray) -> np.ndarray:
    return x

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def activation_backward(name: str | None, pre_activation: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    if name is None or name == "linear":
        return grad_output
    if name == "relu":
        return grad_output * (pre_activation > 0)
    if name == "sigmoid":
        activated = sigmoid(pre_activation)
        return grad_output * activated * (1.0 - activated)
    if name == "tanh":
        activated = tanh(pre_activation)
        return grad_output * (1.0 - activated * activated)
    if name == "softmax":
        probabilities = softmax(pre_activation, axis=-1)
        dot = np.sum(grad_output * probabilities, axis=-1, keepdims=True)
        return probabilities * (grad_output - dot)
    raise ValueError(f"Unsupported activation: {name}")

def get_activation(name: str | None):
    if name is None or name == "linear":
        return linear
    if name == "relu":
        return relu
    if name == "sigmoid":
        return sigmoid
    if name == "tanh":
        return tanh
    if name == "softmax":
        return softmax
    raise ValueError(f"Unsupported activation: {name}")
