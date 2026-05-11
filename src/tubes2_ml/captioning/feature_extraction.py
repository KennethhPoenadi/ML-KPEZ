from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

EncoderName = Literal["inception_v3", "vgg16"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class FeatureExtractionConfig:
    images_dir: Path = Path("data/raw/flickr8k/images")
    output_dir: Path = Path("data/features/captioning")
    encoder_name: EncoderName = "inception_v3"
    batch_size: int = 32
    limit: int | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class FeatureExtractionResult:
    features_path: Path
    image_ids_path: Path
    metadata_path: Path
    num_images: int
    feature_shape: tuple[int, ...]
    encoder_name: str


def list_image_paths(images_dir: str | Path) -> list[Path]:
    directory = Path(images_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Flickr8k image directory not found: {directory}")

    paths = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: path.as_posix())


def image_id_from_path(path: str | Path) -> str:
    return Path(path).stem


def build_encoder(encoder_name: EncoderName):
    import tensorflow as tf

    normalized_name = encoder_name.lower()
    if normalized_name == "inception_v3":
        model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
        target_size = (299, 299)
    elif normalized_name == "vgg16":
        model = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        target_size = (224, 224)
    else:
        raise ValueError("encoder_name must be either 'inception_v3' or 'vgg16'")

    model.trainable = False
    return model, preprocess_input, target_size


def load_image_batch(
    image_paths: Iterable[str | Path],
    target_size: tuple[int, int],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    from PIL import Image

    height, width = target_size
    if height <= 0 or width <= 0:
        raise ValueError("target_size must contain positive integers")

    arrays: list[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            arrays.append(np.asarray(image, dtype=dtype))

    if not arrays:
        return np.empty((0, height, width, 3), dtype=dtype)

    return np.stack(arrays, axis=0)


def extract_flickr8k_features(config: FeatureExtractionConfig) -> FeatureExtractionResult:
    image_paths = list_image_paths(config.images_dir)
    if config.limit is not None:
        if config.limit <= 0:
            raise ValueError("limit must be a positive integer")
        image_paths = image_paths[: config.limit]

    output_dir = Path(config.output_dir)
    features_path = output_dir / "features.npy"
    image_ids_path = output_dir / "image_ids.json"
    metadata_path = output_dir / "metadata.json"

    if features_path.exists() and image_ids_path.exists() and metadata_path.exists() and not config.overwrite:
        features = np.load(features_path)
        return FeatureExtractionResult(
            features_path=features_path,
            image_ids_path=image_ids_path,
            metadata_path=metadata_path,
            num_images=int(features.shape[0]),
            feature_shape=tuple(int(value) for value in features.shape[1:]),
            encoder_name=config.encoder_name,
        )

    if config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not image_paths:
        raise ValueError(f"No images found in {config.images_dir}")

    encoder, preprocess_input, target_size = build_encoder(config.encoder_name)

    feature_batches: list[np.ndarray] = []
    for start in range(0, len(image_paths), config.batch_size):
        batch_paths = image_paths[start : start + config.batch_size]
        batch = load_image_batch(batch_paths, target_size=target_size)
        batch = preprocess_input(batch)
        batch_features = encoder.predict(batch, verbose=0)
        feature_batches.append(np.asarray(batch_features, dtype=np.float32))

    features = np.concatenate(feature_batches, axis=0)
    image_ids = [image_id_from_path(path) for path in image_paths]
    metadata = {
        "encoder_name": config.encoder_name,
        "images_dir": str(config.images_dir),
        "num_images": len(image_ids),
        "feature_shape": list(features.shape[1:]),
        "target_size": list(target_size),
        "batch_size": config.batch_size,
        "features_file": features_path.name,
        "image_ids_file": image_ids_path.name,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_path, features)
    image_ids_path.write_text(json.dumps(image_ids, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return FeatureExtractionResult(
        features_path=features_path,
        image_ids_path=image_ids_path,
        metadata_path=metadata_path,
        num_images=len(image_ids),
        feature_shape=tuple(int(value) for value in features.shape[1:]),
        encoder_name=config.encoder_name,
    )


def load_extracted_features(features_dir: str | Path) -> tuple[np.ndarray, list[str]]:
    directory = Path(features_dir)
    features_path = directory / "features.npy"
    image_ids_path = directory / "image_ids.json"

    if not features_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    if not image_ids_path.is_file():
        raise FileNotFoundError(f"Image id file not found: {image_ids_path}")

    features = np.load(features_path)
    image_ids = json.loads(image_ids_path.read_text(encoding="utf-8"))
    if len(image_ids) != int(features.shape[0]):
        raise ValueError("Number of image ids does not match number of feature rows")

    return features, image_ids


def feature_mapping(features_dir: str | Path) -> dict[str, np.ndarray]:
    features, image_ids = load_extracted_features(features_dir)
    return {image_id: features[index] for index, image_id in enumerate(image_ids)}


def result_to_json(result: FeatureExtractionResult) -> str:
    data = asdict(result)
    for key in ("features_path", "image_ids_path", "metadata_path"):
        data[key] = str(data[key])
    data["feature_shape"] = list(result.feature_shape)
    return json.dumps(data, indent=2)
