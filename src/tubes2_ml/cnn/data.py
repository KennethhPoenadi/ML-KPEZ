from __future__ import annotations
from pathlib import Path
from typing import Iterable, Literal
import numpy as np
from PIL import Image

ColorMode = Literal["rgb", "grayscale"]

def load_image(image_path: str | Path,target_size: tuple[int, int],color_mode: ColorMode = "rgb",normalize: bool = True,dtype: np.dtype = np.float32) -> np.ndarray:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    if color_mode not in {"rgb", "grayscale"}:
        raise ValueError("color_mode must be either 'rgb' or 'grayscale'")

    height, width = target_size
    if height <= 0 or width <= 0:
        raise ValueError("target_size must contain positive integers")

     
    if color_mode == "rgb":
        pil_mode = "RGB"
    else:
        pil_mode = "L"

    with Image.open(path) as image:
        image = image.convert(pil_mode)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=dtype)

    if color_mode == "grayscale":
        array = array[..., np.newaxis]

    if normalize:
        array = array / np.asarray(255.0, dtype=dtype)

    return array.astype(dtype, copy=False)


def load_image_batch(image_paths: Iterable[str | Path],target_size: tuple[int, int],color_mode: ColorMode = "rgb",normalize: bool = True,dtype: np.dtype = np.float32,) -> np.ndarray:
    paths = list(image_paths)
    if not paths:
        if color_mode == "rgb":
            channels = 3    
        else:
            channels = 1

        height, width = target_size
        return np.empty((0, height, width, channels), dtype=dtype)

    images = [
        load_image(image_path=path,target_size=target_size,color_mode=color_mode,normalize=normalize,dtype=dtype)
        for path in paths
    ]

    return np.stack(images, axis=0)
