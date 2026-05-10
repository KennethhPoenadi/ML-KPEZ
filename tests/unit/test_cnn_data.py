import numpy as np
import pytest
from PIL import Image
from tubes2_ml.cnn.data import load_image, load_image_batch

def test_load_image_returns_resized_normalized_rgb_array(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 6), color=(128, 64, 32)).save(image_path)

    image = load_image(image_path, target_size=(4, 5))

    assert image.shape == (4, 5, 3)
    assert image.dtype == np.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0

def test_load_image_grayscale_keeps_channel_dimension(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 6), color=(255, 0, 0)).save(image_path)

    image = load_image(image_path, target_size=(3, 2), color_mode="grayscale")

    assert image.shape == (3, 2, 1)

def test_load_image_batch_returns_nhwc_array(tmp_path):
    image_paths = []
    for index in range(2):
        image_path = tmp_path / f"sample_{index}.jpg"
        Image.new("RGB", (8, 6), color=(index * 50, 64, 32)).save(image_path)
        image_paths.append(image_path)

    batch = load_image_batch(image_paths, target_size=(4, 5))

    assert batch.shape == (2, 4, 5, 3)
    assert batch.dtype == np.float32

def test_load_image_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg", target_size=(4, 5))
