from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tubes2_ml.captioning.decoding import greedy_decode_batch, predict_with_keras_model, predict_with_scratch_captioner
from tubes2_ml.captioning.evaluate import evaluate_caption_predictions, group_reference_tokens
from tubes2_ml.captioning.feature_extraction import feature_mapping
from tubes2_ml.captioning.inference import infer_decoder_kind, infer_injection_mode, load_vocabulary
from tubes2_ml.captioning.preprocessing import load_caption_records
from tubes2_ml.captioning.train import load_processed_split
from tubes2_ml.scratch.models.lstm_captioner import build_scratch_lstm_captioner_from_keras
from tubes2_ml.scratch.models.rnn_captioner import build_scratch_rnn_captioner_from_keras


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = str(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def parse_max_lengths(value: str | None, default_length: int) -> list[int]:
    if value in {None, ""}:
        return [default_length]
    lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("max caption lengths must be positive integers")
    return lengths


def discover_model_paths(models_dir: str | Path) -> list[Path]:
    directory = Path(models_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Model directory not found: {directory}")
    return sorted(directory.glob("*.keras"))


def scratch_predictor_from_keras_model(keras_model):
    decoder_kind = infer_decoder_kind(keras_model)
    if decoder_kind == "rnn":
        captioner = build_scratch_rnn_captioner_from_keras(keras_model)
    elif decoder_kind == "lstm":
        captioner = build_scratch_lstm_captioner_from_keras(keras_model)
    else:
        raise ValueError(f"Unsupported decoder kind: {decoder_kind}")
    return predict_with_scratch_captioner(captioner)


def generate_predictions(
    predict_probs,
    image_ids: list[str],
    features: dict[str, np.ndarray],
    vocabulary,
    max_caption_length: int,
) -> dict[str, str]:
    missing: list[str] = []
    for image_id in image_ids:
        if image_id not in features:
            missing.append(image_id)

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing extracted features for {len(missing)} image ids. First missing ids: {preview}")

    feature_batch = np.stack([features[image_id] for image_id in image_ids], axis=0)
    batch_token_ids = greedy_decode_batch(
        predict_probs,
        feature_batch,
        vocabulary,
        max_caption_length=max_caption_length,
    )
    predictions = {
        image_id: vocabulary.ids_to_caption(token_ids)
        for image_id, token_ids in zip(image_ids, batch_token_ids)
    }
    return predictions


def evaluate_one_model(
    model_path: Path,
    backends: list[str],
    image_ids: list[str],
    features: dict[str, np.ndarray],
    vocabulary,
    references: dict[str, list[list[str]]],
    max_caption_lengths: list[int],
    predictions_dir: Path,
) -> list[dict[str, Any]]:
    import tensorflow as tf

    keras_model = tf.keras.models.load_model(model_path)
    rows: list[dict[str, Any]] = []
    predictors = {"keras": predict_with_keras_model(keras_model)}
    if "scratch" in backends:
        if infer_injection_mode(keras_model) == "pre":
            predictors["scratch"] = scratch_predictor_from_keras_model(keras_model)
        else:
            print(f"Skipping scratch backend for init-inject model: {model_path.name}")

    for backend in backends:
        if backend not in predictors:
            if backend not in {"keras", "scratch"}:
                raise ValueError("backend must be 'keras' or 'scratch'")
            continue
        for max_caption_length in max_caption_lengths:
            start_time = time.perf_counter()
            predictions = generate_predictions(
                predictors[backend],
                image_ids=image_ids,
                features=features,
                vocabulary=vocabulary,
                max_caption_length=max_caption_length,
            )
            elapsed_seconds = time.perf_counter() - start_time
            metrics = evaluate_caption_predictions(predictions, references)

            predictions_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = predictions_dir / f"{model_path.stem}_{backend}_maxlen{max_caption_length}.json"
            predictions_payload = [
                {
                    "image_id": image_id,
                    "caption": predictions[image_id],
                    "references": [" ".join(tokens) for tokens in references.get(image_id, [])],
                }
                for image_id in image_ids
            ]
            predictions_path.write_text(json.dumps(predictions_payload, indent=2), encoding="utf-8")

            rows.append(
                {
                    "model": model_path.stem,
                    "backend": backend,
                    "max_caption_length": max_caption_length,
                    "bleu4": metrics["bleu4"],
                    "meteor": metrics["meteor"],
                    "num_predictions": int(metrics["num_predictions"]),
                    "execution_time_seconds": elapsed_seconds,
                    "seconds_per_image": elapsed_seconds / max(1, len(image_ids)),
                    "predictions_path": str(predictions_path),
                }
            )
    return rows


def write_rows(rows: list[dict[str, Any]], output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def decoder_family(model_name: str) -> str:
    if model_name.startswith("rnn_") or model_name.startswith("init_rnn_"):
        return "rnn"
    if model_name.startswith("lstm_") or model_name.startswith("init_lstm_"):
        return "lstm"
    return "unknown"


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row["bleu4"]), float(row["meteor"])))


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "best_overall": best_row(rows),
        "best_by_decoder": {},
        "best_by_backend": {},
        "best_by_max_caption_length": {},
    }

    for decoder in sorted({decoder_family(str(row["model"])) for row in rows}):
        subset = [row for row in rows if decoder_family(str(row["model"])) == decoder]
        summary["best_by_decoder"][decoder] = best_row(subset)

    for backend in sorted({str(row["backend"]) for row in rows}):
        subset = [row for row in rows if str(row["backend"]) == backend]
        summary["best_by_backend"][backend] = best_row(subset)

    for max_length in sorted({int(row["max_caption_length"]) for row in rows}):
        subset = [row for row in rows if int(row["max_caption_length"]) == max_length]
        summary["best_by_max_caption_length"][str(max_length)] = best_row(subset)

    return summary


def write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summarize_results(rows), indent=2), encoding="utf-8")


def export_qualitative_samples(
    rows: list[dict[str, Any]],
    output_path: Path,
    sample_count: int = 10,
) -> None:
    best = best_row(rows)
    if best is None:
        output_path.write_text("[]", encoding="utf-8")
        return

    predictions_path = Path(str(best["predictions_path"]))
    if not predictions_path.is_file():
        output_path.write_text("[]", encoding="utf-8")
        return

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    samples = predictions[:sample_count]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "source_result": best,
                "samples": samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained RNN/LSTM captioning experiments.")
    parser.add_argument("--models-dir", default="models/keras/captioning")
    parser.add_argument("--processed-dir", default="data/processed/captioning")
    parser.add_argument("--features-dir", default="data/features/captioning")
    parser.add_argument("--captions-path", default="data/raw/flickr8k/captions/captions.txt")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--backends", default="keras", help="Comma-separated: keras,scratch")
    parser.add_argument("--max-caption-lengths", default=None, help="Comma-separated, e.g. 10,20,38")
    parser.add_argument("--output-csv", default="artifacts/experiments/captioning/evaluation_results.csv")
    parser.add_argument("--output-json", default="artifacts/experiments/captioning/evaluation_results.json")
    parser.add_argument("--summary-json", default="artifacts/experiments/captioning/evaluation_summary.json")
    parser.add_argument("--qualitative-json", default="artifacts/predictions/captioning/qualitative_samples.json")
    parser.add_argument("--qualitative-samples", type=int, default=10)
    parser.add_argument("--predictions-dir", default="artifacts/predictions/captioning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = resolve_project_path(args.models_dir)
    processed_dir = resolve_project_path(args.processed_dir)
    features_dir = resolve_project_path(args.features_dir)
    captions_path = resolve_project_path(args.captions_path)
    output_csv = resolve_project_path(args.output_csv)
    output_json = resolve_project_path(args.output_json)
    summary_json = resolve_project_path(args.summary_json)
    qualitative_json = resolve_project_path(args.qualitative_json)
    predictions_dir = resolve_project_path(args.predictions_dir)

    split_data = load_processed_split(processed_dir, args.split)
    image_ids = unique_preserve_order(split_data["image_ids"])
    if args.limit_images is not None:
        if args.limit_images <= 0:
            raise ValueError("limit_images must be positive")
        image_ids = image_ids[: args.limit_images]

    vocabulary = load_vocabulary(processed_dir / "vocabulary.json")
    max_caption_lengths = parse_max_lengths(args.max_caption_lengths, int(split_data["input_sequences"].shape[1]))
    features = feature_mapping(features_dir)
    references = group_reference_tokens(load_caption_records(captions_path))
    backends = [backend.strip() for backend in args.backends.split(",") if backend.strip()]
    model_paths = discover_model_paths(models_dir)

    rows: list[dict[str, Any]] = []
    for model_path in model_paths:
        print(f"Evaluating {model_path.name} on {len(image_ids)} images...")
        rows.extend(
            evaluate_one_model(
                model_path=model_path,
                backends=backends,
                image_ids=image_ids,
                features=features,
                vocabulary=vocabulary,
                references=references,
                max_caption_lengths=max_caption_lengths,
                predictions_dir=predictions_dir,
            )
        )

    write_rows(rows, output_csv=output_csv, output_json=output_json)
    write_summary(rows, summary_json)
    export_qualitative_samples(rows, qualitative_json, sample_count=args.qualitative_samples)
    print(f"Wrote {len(rows)} evaluation rows to {output_csv}")
    print(f"Wrote evaluation summary to {summary_json}")
    print(f"Wrote qualitative samples to {qualitative_json}")


if __name__ == "__main__":
    main()
