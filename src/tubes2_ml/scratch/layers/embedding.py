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
        return self.weights[ids]
