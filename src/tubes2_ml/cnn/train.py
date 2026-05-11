from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import tensorflow as tf
from tubes2_ml.cnn.models import SharedConvCNNConfig, build_shared_conv_cnn
from tubes2_ml.evaluation.cnn_metrics import macro_f1_score

@dataclass(frozen=True)
class CNNTrainingConfig:
    train_dir: str | Path = "data/raw/intel_image_classification/seg_train/seg_train"  # intel train folder
    validation_dir: str | Path | None = None  # use validation_split when this is None
    output_dir: str | Path = "models/keras/cnn"  # saved Keras models and weights <- for the path yaw
    history_dir: str | Path = "artifacts/experiments/cnn"  # training logs and metadata
    image_size: tuple[int, int] = (96, 96)  # image resize target as (height, width)
    batch_size: int = 64  # number of images per training batch
    epochs: int = 5  # number of full passes over the training set
    validation_split: float = 0.2  # fraction of train_dir used for validation
    seed: int = 42
    save_format: str = "keras"  # save full model as .keras when enabled
    early_stopping_patience: int | None = 1

def _make_dataset_from_directory(directory: str | Path,image_size: tuple[int, int],batch_size: int,seed: int,subset: str | None = None,validation_split: float | None = None,shuffle: bool = True):
    kwargs: dict[str, Any] = {
        "directory": str(directory),
        "labels": "inferred",
        "label_mode": "int",
        "color_mode": "rgb",
        "batch_size": batch_size,
        "image_size": image_size,
        "shuffle": shuffle,
        "seed": seed,
    }

    if subset is not None:
        kwargs["subset"] = subset
        kwargs["validation_split"] = validation_split

    dataset = tf.keras.utils.image_dataset_from_directory(**kwargs)
    class_names = list(dataset.class_names)
    dataset = dataset.map(lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels))

    return dataset, class_names


def build_intel_datasets(config: CNNTrainingConfig):
    train_dir = Path(config.train_dir)

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if config.validation_dir is not None:
        train_ds, class_names = _make_dataset_from_directory(
            train_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
            seed=config.seed,
            shuffle=True,
        )
        val_ds, _ = _make_dataset_from_directory(
            config.validation_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
            seed=config.seed,
            shuffle=False,
        )
    else:
        train_ds, class_names = _make_dataset_from_directory(
            train_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
            seed=config.seed,
            subset="training",
            validation_split=config.validation_split,
            shuffle=True,
        )
        val_ds, _ = _make_dataset_from_directory(
            train_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
            seed=config.seed,
            subset="validation",
            validation_split=config.validation_split,
            shuffle=True,
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names


def _jsonable_history(history: dict[str, list[Any]]) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.items()}

def train_shared_conv_cnn(model_config: SharedConvCNNConfig,training_config: CNNTrainingConfig,callbacks: list[Any] | None = None):

    train_ds, val_ds, class_names = build_intel_datasets(training_config)

    output_dir = Path(training_config.output_dir)
    history_dir = Path(training_config.history_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    model = build_shared_conv_cnn(model_config)
    run_name = model_config.name
    weights_path = output_dir / f"{run_name}.weights.h5"

    default_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
        ),
        tf.keras.callbacks.CSVLogger(str(history_dir / f"{run_name}.csv")),
    ]

    if training_config.early_stopping_patience is not None:
        default_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=training_config.early_stopping_patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_config.epochs,
        callbacks=[*default_callbacks, *(callbacks or [])],
    )

    if weights_path.exists():
        model.load_weights(weights_path)

    val_pred, val_true = predict_dataset(model, val_ds)
    val_macro_f1 = macro_f1_score(val_true, val_pred, num_classes=len(class_names))

    model_path = output_dir / f"{run_name}.keras"

    if training_config.save_format == "keras":
        model.save(model_path)

    metadata = {
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "class_names": class_names,
        "history": _jsonable_history(history.history),
        "metrics": {
            "validation_macro_f1": val_macro_f1,
        },
        "artifacts": {
            "weights_path": str(weights_path),
            "model_path": str(model_path) if training_config.save_format == "keras" else None,
        },
    }
    metadata_path = history_dir / f"{run_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return model, history, metadata

def predict_dataset(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_pred.append(np.argmax(probabilities, axis=1))
        y_true.append(np.asarray(labels))

    return np.concatenate(y_pred, axis=0), np.concatenate(y_true, axis=0)
