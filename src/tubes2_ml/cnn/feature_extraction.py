from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterable
import numpy as np
from tubes2_ml.cnn.data import ColorMode, load_image_batch

PreprocessFn = Callable[[np.ndarray], np.ndarray]

def extract_features_to_npy(image_paths: Iterable[str | Path],encoder,output_path: str | Path,target_size: tuple[int, int],batch_size: int = 32,color_mode: ColorMode = "rgb",preprocess_fn: PreprocessFn | None = None,overwrite: bool = False,dtype: np.dtype = np.float32) -> np.ndarray:
    paths = list(image_paths)
    destination = Path(output_path)

    if destination.exists() and not overwrite:
        return np.load(destination)

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    if hasattr(encoder, "trainable"):
        encoder.trainable = False

    features: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = load_image_batch(batch_paths,target_size=target_size,color_mode=color_mode,normalize=True,dtype=dtype)
        if preprocess_fn is not None:
            batch = preprocess_fn(batch)

        batch_features = encoder.predict(batch, verbose=0)
        features.append(np.asarray(batch_features, dtype=dtype))

    if features:
        feature_array = np.concatenate(features, axis=0)
    else:
        feature_array = np.empty((0,), dtype=dtype)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, feature_array)
    return feature_array