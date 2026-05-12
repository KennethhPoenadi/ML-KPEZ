from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tubes2_ml.cnn.evaluate import build_intel_test_dataset
from tubes2_ml.evaluation.cnn_metrics import macro_f1_score
from tubes2_ml.scratch.models.cnn_classifier import build_scratch_cnn_from_keras


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_cnn_runs(experiments_dir: str | Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for metadata_path in sorted(Path(experiments_dir).glob("*.json")):
        if metadata_path.name in {"best_model_evaluation.json", "summary.json"}:
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metric = payload.get("metrics", {}).get("validation_macro_f1")
        if metric is None:
            continue
        payload["_metadata_path"] = str(metadata_path)
        runs.append(payload)
    if not runs:
        raise FileNotFoundError(f"No CNN experiment metadata found in {experiments_dir}")
    return runs


def choose_best_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return max(runs, key=lambda run: float(run["metrics"]["validation_macro_f1"]))


def collect_keras_predictions(keras_model, dataset, max_batches: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for batch_index, (images, labels) in enumerate(dataset):
        if max_batches is not None and batch_index >= max_batches:
            break
        probabilities = keras_model.predict(images, verbose=0)
        y_pred.append(np.argmax(probabilities, axis=-1))
        y_true.append(np.asarray(labels))
    return np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)


def collect_scratch_predictions(scratch_model, dataset, max_batches: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for batch_index, (images, labels) in enumerate(dataset):
        if max_batches is not None and batch_index >= max_batches:
            break
        y_pred.append(scratch_model.predict(np.asarray(images)))
        y_true.append(np.asarray(labels))
    return np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)


def parameter_summary(keras_model, scratch_shared, scratch_non_shared) -> dict[str, int]:
    return {
        "keras_trainable_and_non_trainable": int(keras_model.count_params()),
        "scratch_shared": int(scratch_shared.count_parameters()),
        "scratch_non_shared": int(scratch_non_shared.count_parameters()),
    }


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float | int]:
    return {
        "macro_f1": macro_f1_score(y_true, y_pred, num_classes=num_classes),
        "num_samples": int(y_true.shape[0]),
    }


def agreement(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("Prediction arrays must have the same shape for agreement")
    return float(np.mean(reference == candidate))


def save_predictions(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def evaluate_best_cnn(
    experiments_dir: Path,
    test_dir: Path,
    output_json: Path,
    predictions_path: Path,
    batch_size: int,
    max_batches: int | None,
    scratch_max_batches: int | None,
) -> dict[str, Any]:
    import tensorflow as tf

    runs = load_cnn_runs(experiments_dir)
    best_run = choose_best_run(runs)
    model_path = Path(best_run["artifacts"]["model_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Best Keras model file not found: {model_path}")

    image_size = tuple(int(value) for value in best_run["training_config"].get("image_size", [96, 96]))
    test_ds, class_names = build_intel_test_dataset(test_dir=test_dir, image_size=image_size, batch_size=batch_size)
    num_classes = len(class_names)

    keras_model = tf.keras.models.load_model(model_path)
    input_shape = tuple(int(dim) for dim in keras_model.input_shape[1:])

    started = time.perf_counter()
    keras_true, keras_pred = collect_keras_predictions(keras_model, test_ds, max_batches=max_batches)
    keras_seconds = time.perf_counter() - started

    scratch_shared = build_scratch_cnn_from_keras(
        keras_model,
        replace_conv_with_local=False,
        input_shape=input_shape,
    )
    scratch_non_shared = build_scratch_cnn_from_keras(
        keras_model,
        replace_conv_with_local=True,
        input_shape=input_shape,
    )

    started = time.perf_counter()
    scratch_shared_true, scratch_shared_pred = collect_scratch_predictions(
        scratch_shared,
        test_ds,
        max_batches=scratch_max_batches,
    )
    scratch_shared_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scratch_non_shared_true, scratch_non_shared_pred = collect_scratch_predictions(
        scratch_non_shared,
        test_ds,
        max_batches=scratch_max_batches,
    )
    scratch_non_shared_seconds = time.perf_counter() - started

    params = parameter_summary(keras_model, scratch_shared, scratch_non_shared)
    shared_subset_count = int(scratch_shared_pred.shape[0])
    non_shared_subset_count = int(scratch_non_shared_pred.shape[0])
    keras_shared_prefix = keras_pred[:shared_subset_count]
    keras_non_shared_prefix = keras_pred[:non_shared_subset_count]
    payload: dict[str, Any] = {
        "best_experiment": best_run["model_config"]["name"],
        "best_validation_macro_f1": float(best_run["metrics"]["validation_macro_f1"]),
        "metadata_path": best_run["_metadata_path"],
        "model_path": str(model_path),
        "class_names": class_names,
        "test_dir": str(test_dir),
        "batch_size": batch_size,
        "max_batches": max_batches,
        "scratch_max_batches": scratch_max_batches,
        "metrics": {
            "keras": {
                **prediction_metrics(keras_true, keras_pred, num_classes=num_classes),
                "execution_time_seconds": keras_seconds,
            },
            "scratch_shared": {
                **prediction_metrics(scratch_shared_true, scratch_shared_pred, num_classes=num_classes),
                "execution_time_seconds": scratch_shared_seconds,
                "prediction_agreement_with_keras_same_prefix": agreement(
                    keras_shared_prefix,
                    scratch_shared_pred,
                ),
            },
            "scratch_non_shared": {
                **prediction_metrics(scratch_non_shared_true, scratch_non_shared_pred, num_classes=num_classes),
                "execution_time_seconds": scratch_non_shared_seconds,
                "prediction_agreement_with_keras_same_prefix": agreement(
                    keras_non_shared_prefix,
                    scratch_non_shared_pred,
                ),
            },
        },
        "parameter_counts": params,
        "parameter_ratio_non_shared_vs_shared": (
            float(params["scratch_non_shared"] / params["scratch_shared"])
            if params["scratch_shared"] else None
        ),
        "notes": [
            "scratch_non_shared replaces every Conv2D layer with LocallyConnected2D using repeated trained Conv2D weights.",
            "When scratch_max_batches is not null, scratch metrics are a subset smoke/evaluation run.",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_predictions(
        predictions_path,
        keras_y_true=keras_true,
        keras_y_pred=keras_pred,
        scratch_shared_y_true=scratch_shared_true,
        scratch_shared_y_pred=scratch_shared_pred,
        scratch_non_shared_y_true=scratch_non_shared_true,
        scratch_non_shared_y_pred=scratch_non_shared_pred,
    )
    return payload


def parse_optional_int(value: str | None) -> int | None:
    if value in {None, "", "none", "None", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("batch limits must be positive integers")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best CNN experiment on the Intel test split.")
    parser.add_argument("--experiments-dir", default="artifacts/experiments/cnn")
    parser.add_argument("--test-dir", default="data/raw/intel_image_classification/seg_test/seg_test")
    parser.add_argument("--output-json", default="artifacts/experiments/cnn/best_model_evaluation.json")
    parser.add_argument("--predictions-path", default="artifacts/predictions/cnn/best_model_predictions.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", default=None, help="Limit Keras test batches; default full test split.")
    parser.add_argument("--scratch-max-batches", default=None, help="Limit scratch test batches; default full test split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_best_cnn(
        experiments_dir=resolve_project_path(args.experiments_dir),
        test_dir=resolve_project_path(args.test_dir),
        output_json=resolve_project_path(args.output_json),
        predictions_path=resolve_project_path(args.predictions_path),
        batch_size=args.batch_size,
        max_batches=parse_optional_int(args.max_batches),
        scratch_max_batches=parse_optional_int(args.scratch_max_batches),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
