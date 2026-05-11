from __future__ import annotations
from pathlib import Path
import numpy as np
import tensorflow as tf

from tubes2_ml.evaluation.cnn_metrics import macro_f1_score


def build_intel_test_dataset(
    test_dir: str | Path = "data/raw/intel_image_classification/seg_test/seg_test",
    image_size: tuple[int, int] = (96, 96),
    batch_size: int = 64,
):
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    class_names = list(dataset.class_names)
    dataset = dataset.map(lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels))
    return dataset.prefetch(tf.data.AUTOTUNE), class_names


def evaluate_keras_model(model, dataset, num_classes: int, max_batches: int | None = None) -> dict[str, float]:
    y_true, y_pred = [], []

    for batch_index, (images, labels) in enumerate(dataset):
        if max_batches is not None and batch_index >= max_batches:
            break
        probabilities = model.predict(images, verbose=0)
        y_pred.append(np.argmax(probabilities, axis=-1))
        y_true.append(np.asarray(labels))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {"macro_f1": macro_f1_score(y_true, y_pred, num_classes=num_classes)}


def evaluate_scratch_model(model, dataset, num_classes: int, max_batches: int | None = None) -> dict[str, float]:
    y_true, y_pred = [], []

    for batch_index, (images, labels) in enumerate(dataset):
        if max_batches is not None and batch_index >= max_batches:
            break
        y_pred.append(model.predict(np.asarray(images)))
        y_true.append(np.asarray(labels))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {"macro_f1": macro_f1_score(y_true, y_pred, num_classes=num_classes)}
