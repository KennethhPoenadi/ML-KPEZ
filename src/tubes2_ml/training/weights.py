from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


def _normalize_arrays(weights: Any) -> Dict[str, np.ndarray]:
    if isinstance(weights, dict):
        return {str(k): np.asarray(v) for k, v in weights.items()}
    if isinstance(weights, (list, tuple)):
        return {f"arr_{idx}": np.asarray(value) for idx, value in enumerate(weights)}
    if isinstance(weights, np.ndarray):
        return {"arr_0": weights}
    raise TypeError("Unsupported weights type for numpy serialization")


def save_weights(path: str | Path, weights: Any, metadata: Dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(weights, "save_weights"):
        weights.save_weights(str(path))
        return path

    arrays = _normalize_arrays(weights)
    np.savez_compressed(path, **arrays)

    if metadata:
        sidecar = path.with_suffix(path.suffix + ".json")
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return path


def load_weights(path: str | Path, target: Any | None = None):
    path = Path(path)

    if target is not None and hasattr(target, "load_weights"):
        target.load_weights(str(path))
        return target

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    raise ValueError("Unsupported weight format. Provide a target with load_weights or a .npz file.")


def list_weight_files(directory: str | Path, patterns: Iterable[str] | None = None) -> list[Path]:
    directory = Path(directory)
    if patterns is None:
        patterns = ["*.npz", "*.weights.h5", "*.h5", "*.keras"]

    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    return files
