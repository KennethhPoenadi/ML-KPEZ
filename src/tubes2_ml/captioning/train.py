from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tubes2_ml.captioning.feature_extraction import load_extracted_features
from tubes2_ml.captioning.models import CaptionDecoderConfig, DecoderType, build_preinject_decoder


@dataclass(frozen=True)
class CaptionTrainingConfig:
    processed_dir: Path = Path("data/processed/captioning")
    features_dir: Path = Path("data/features/captioning")
    output_dir: Path = Path("models/keras/captioning")
    history_dir: Path = Path("artifacts/experiments/captioning")
    batch_size: int = 64
    epochs: int = 10
    seed: int = 42


@dataclass(frozen=True)
class CaptionTrainingResult:
    model_path: Path
    history_path: Path
    metadata_path: Path
    decoder_type: str
    num_recurrent_layers: int
    hidden_units: int
    best_validation_loss: float


def load_vocabulary(processed_dir: str | Path) -> dict:
    vocabulary_path = Path(processed_dir) / "vocabulary.json"
    if not vocabulary_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {vocabulary_path}")
    return json.loads(vocabulary_path.read_text(encoding="utf-8"))


def load_processed_split(processed_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    split_path = Path(processed_dir) / f"{split}.npz"
    if not split_path.is_file():
        raise FileNotFoundError(f"Processed split file not found: {split_path}")
    data = np.load(split_path)
    return {key: data[key] for key in data.files}


def build_feature_lookup(features_dir: str | Path) -> dict[str, np.ndarray]:
    features, image_ids = load_extracted_features(features_dir)
    return {image_id: features[index].astype(np.float32, copy=False) for index, image_id in enumerate(image_ids)}


def gather_features(sample_image_ids: np.ndarray, feature_lookup: dict[str, np.ndarray]) -> np.ndarray:
    missing = [image_id for image_id in sample_image_ids if str(image_id) not in feature_lookup]
    if missing:
        preview = ", ".join(str(image_id) for image_id in missing[:5])
        raise ValueError(f"Missing extracted features for {len(missing)} samples. First missing ids: {preview}")
    return np.stack([feature_lookup[str(image_id)] for image_id in sample_image_ids], axis=0).astype(np.float32)


def make_tf_dataset(
    features: np.ndarray,
    input_sequences: np.ndarray,
    target_sequences: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    import tensorflow as tf

    targets = target_sequences[..., np.newaxis]
    sample_weights = (target_sequences != 0).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(((features, input_sequences), targets, sample_weights))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(input_sequences), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def experiment_name(config: CaptionDecoderConfig) -> str:
    return f"{config.decoder_type}_layers{config.num_recurrent_layers}_hidden{config.hidden_units}"


def train_caption_decoder(
    model_config: CaptionDecoderConfig,
    training_config: CaptionTrainingConfig,
) -> CaptionTrainingResult:
    import tensorflow as tf

    if training_config.batch_size <= 0 or training_config.epochs <= 0:
        raise ValueError("batch_size and epochs must be positive integers")

    tf.keras.utils.set_random_seed(training_config.seed)

    vocabulary = load_vocabulary(training_config.processed_dir)
    split_train = load_processed_split(training_config.processed_dir, "train")
    split_validation = load_processed_split(training_config.processed_dir, "validation")
    feature_lookup = build_feature_lookup(training_config.features_dir)

    train_features = gather_features(split_train["image_ids"], feature_lookup)
    validation_features = gather_features(split_validation["image_ids"], feature_lookup)

    inferred_config = CaptionDecoderConfig(
        vocab_size=len(vocabulary["word_to_id"]),
        feature_dim=int(train_features.shape[1]),
        max_caption_length=int(split_train["input_sequences"].shape[1]),
        embed_dim=model_config.embed_dim,
        hidden_units=model_config.hidden_units,
        num_recurrent_layers=model_config.num_recurrent_layers,
        dropout_rate=model_config.dropout_rate,
        learning_rate=model_config.learning_rate,
        decoder_type=model_config.decoder_type,
        name=model_config.name,
    )
    model = build_preinject_decoder(inferred_config)

    train_dataset = make_tf_dataset(
        train_features,
        split_train["input_sequences"],
        split_train["target_sequences"],
        training_config.batch_size,
        shuffle=True,
        seed=training_config.seed,
    )
    validation_dataset = make_tf_dataset(
        validation_features,
        split_validation["input_sequences"],
        split_validation["target_sequences"],
        training_config.batch_size,
        shuffle=False,
        seed=training_config.seed,
    )

    name = experiment_name(inferred_config)
    output_dir = Path(training_config.output_dir)
    history_dir = Path(training_config.history_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{name}.keras"
    history_path = history_dir / f"{name}_history.json"
    metadata_path = history_dir / f"{name}_metadata.json"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=training_config.epochs,
        callbacks=callbacks,
    )

    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    validation_losses = history_payload.get("val_loss", [])
    best_validation_loss = min(validation_losses) if validation_losses else float("nan")
    metadata = {
        "model_config": asdict(inferred_config),
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(training_config).items()
        },
        "model_path": str(model_path),
        "history_path": str(history_path),
        "best_validation_loss": best_validation_loss,
        "train_samples": int(split_train["input_sequences"].shape[0]),
        "validation_samples": int(split_validation["input_sequences"].shape[0]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return CaptionTrainingResult(
        model_path=model_path,
        history_path=history_path,
        metadata_path=metadata_path,
        decoder_type=inferred_config.decoder_type,
        num_recurrent_layers=inferred_config.num_recurrent_layers,
        hidden_units=inferred_config.hidden_units,
        best_validation_loss=float(best_validation_loss),
    )


def make_base_model_config(decoder_type: DecoderType, num_recurrent_layers: int, hidden_units: int) -> CaptionDecoderConfig:
    return CaptionDecoderConfig(
        vocab_size=1,
        feature_dim=1,
        max_caption_length=1,
        decoder_type=decoder_type,
        num_recurrent_layers=num_recurrent_layers,
        hidden_units=hidden_units,
        name=f"{decoder_type}_layers{num_recurrent_layers}_hidden{hidden_units}",
    )
