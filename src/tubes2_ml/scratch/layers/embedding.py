from __future__ import annotations

import numpy as np


class Embedding:
    def __init__(self, weights: np.ndarray | None = None):
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)

    def load_keras_weights(self, weights: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
        if len(weights) != 1:
            raise ValueError("Embedding weights must be [embeddings]")
        self.weights = np.asarray(weights[0], dtype=np.float32)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Embedding weights are not loaded")

        ids = np.asarray(token_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.weights.shape[0]):
            raise ValueError("Token id out of embedding vocabulary range")
        self._ids = ids
        return self.weights[ids]

    def backward(self, grad_output: np.ndarray) -> None:
        if self.weights is None:
            raise ValueError("Embedding weights are not loaded")
        if not hasattr(self, "_ids"):
            raise ValueError("Embedding backward called before forward")

        self.grad_weights = np.zeros_like(self.weights)
        np.add.at(self.grad_weights, self._ids, np.asarray(grad_output, dtype=np.float32))
        return None
