from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        raise ValueError("Cannot compute macro F1 on empty arrays")

    labels = list(range(num_classes)) if num_classes is not None else None
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
