import numpy as np
from PIL import Image
from tubes2_ml.cnn.feature_extraction import extract_features_to_npy

class FakeEncoder:
    def __init__(self):
        self.trainable = True
        self.batch_shapes = []

    def predict(self, batch, verbose=0):
        self.batch_shapes.append(batch.shape)
        means = batch.mean(axis=(1, 2, 3), keepdims=False)
        maxima = batch.max(axis=(1, 2, 3))
        return np.stack([means, maxima], axis=1)

def test_extract_features_to_npy_saves_features_and_freezes_encoder(tmp_path):
    image_paths = []
    for index, color in enumerate([(0, 0, 0), (128, 64, 32), (255, 255, 255)]):
        image_path = tmp_path / f"image_{index}.jpg"
        Image.new("RGB", (8, 6), color=color).save(image_path)
        image_paths.append(image_path)

    encoder = FakeEncoder()
    output_path = tmp_path / "features" / "intel_features.npy"

    features = extract_features_to_npy(
        image_paths=image_paths,
        encoder=encoder,
        output_path=output_path,
        target_size=(4, 5),
        batch_size=2,
    )

    assert encoder.trainable is False
    assert encoder.batch_shapes == [(2, 4, 5, 3), (1, 4, 5, 3)]
    assert features.shape == (3, 2)
    assert output_path.is_file()
    np.testing.assert_allclose(np.load(output_path), features)

def test_extract_features_to_npy_uses_cache_when_output_exists(tmp_path):
    output_path = tmp_path / "features.npy"
    cached = np.array([[1.0, 2.0]], dtype=np.float32)
    np.save(output_path, cached)

    encoder = FakeEncoder()
    features = extract_features_to_npy(
        image_paths=[],
        encoder=encoder,
        output_path=output_path,
        target_size=(4, 5),
    )

    np.testing.assert_allclose(features, cached)
    assert encoder.batch_shapes == []
