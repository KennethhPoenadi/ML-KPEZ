from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from tubes2_ml.captioning.preprocessing import CaptionRecord, load_caption_records, tokenize_caption


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def sentence_bleu4(
    references: list[list[str]],
    candidate: list[str],
    max_n: int = 4,
    smoothing: float = 1.0,
) -> float:
    if not candidate:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        candidate_ngrams = ngrams(candidate, n)
        if not candidate_ngrams:
            precisions.append(smoothing / smoothing)
            continue

        clipped_count = 0
        for gram, count in candidate_ngrams.items():
            max_ref_count = max((ngrams(reference, n).get(gram, 0) for reference in references), default=0)
            clipped_count += min(count, max_ref_count)

        precisions.append((clipped_count + smoothing) / (sum(candidate_ngrams.values()) + smoothing))

    reference_lengths = [len(reference) for reference in references]
    closest_ref_length = min(reference_lengths, key=lambda length: (abs(length - len(candidate)), length))
    brevity_penalty = 1.0 if len(candidate) > closest_ref_length else math.exp(1 - closest_ref_length / len(candidate))
    return float(brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / max_n))


def meteor_score_fallback(references: list[list[str]], candidate: list[str]) -> float:
    if not candidate:
        return 0.0

    best_score = 0.0
    candidate_counts = Counter(candidate)
    for reference in references:
        reference_counts = Counter(reference)
        matches = sum(min(count, reference_counts[token]) for token, count in candidate_counts.items())
        if matches == 0:
            continue
        precision = matches / len(candidate)
        recall = matches / len(reference) if reference else 0.0
        if precision + recall == 0:
            continue
        score = (10 * precision * recall) / (recall + 9 * precision)
        best_score = max(best_score, score)
    return float(best_score)


def sentence_meteor(references: list[list[str]], candidate: list[str]) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score
    except Exception:
        return meteor_score_fallback(references, candidate)

    try:
        return float(meteor_score(references, candidate))
    except Exception:
        return meteor_score_fallback(references, candidate)


def group_reference_tokens(records: Iterable[CaptionRecord]) -> dict[str, list[list[str]]]:
    grouped: dict[str, list[list[str]]] = {}
    for record in records:
        grouped.setdefault(record.image_id, []).append(tokenize_caption(record.caption))
    return grouped


def evaluate_caption_predictions(
    predictions: dict[str, str],
    references: dict[str, list[list[str]]],
) -> dict[str, float]:
    bleu_scores: list[float] = []
    meteor_scores: list[float] = []

    for image_id, caption in predictions.items():
        if image_id not in references:
            continue
        candidate = tokenize_caption(caption)
        reference_tokens = references[image_id]
        bleu_scores.append(sentence_bleu4(reference_tokens, candidate))
        meteor_scores.append(sentence_meteor(reference_tokens, candidate))

    if not bleu_scores:
        return {"bleu4": 0.0, "meteor": 0.0, "num_predictions": 0.0}

    return {
        "bleu4": float(np.mean(bleu_scores)),
        "meteor": float(np.mean(meteor_scores)),
        "num_predictions": float(len(bleu_scores)),
    }


def load_prediction_json(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if all(isinstance(value, str) for value in payload.values()):
            return {str(key): str(value) for key, value in payload.items()}
        if "predictions" in payload:
            payload = payload["predictions"]

    predictions: dict[str, str] = {}
    for item in payload:
        predictions[str(item["image_id"])] = str(item["caption"])
    return predictions


def evaluate_prediction_file(predictions_path: str | Path, captions_path: str | Path) -> dict[str, float]:
    predictions = load_prediction_json(predictions_path)
    references = group_reference_tokens(load_caption_records(captions_path))
    return evaluate_caption_predictions(predictions, references)
