from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from tubes2_ml.captioning.decoding import (
    CaptionVocabulary,
    beam_search_decode,
    greedy_decode,
    predict_with_keras_model,
    predict_with_scratch_captioner,
)
from tubes2_ml.captioning.feature_extraction import feature_mapping
from tubes2_ml.scratch.models.lstm_captioner import build_scratch_lstm_captioner_from_keras
from tubes2_ml.scratch.models.rnn_captioner import build_scratch_rnn_captioner_from_keras

Backend = Literal["keras", "scratch"]
DecoderKind = Literal["rnn", "lstm"]
SearchStrategy = Literal["greedy", "beam"]


def load_vocabulary(vocabulary_path: str | Path) -> CaptionVocabulary:
    payload = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))
    special = payload.get("special_tokens", {})
    return CaptionVocabulary(
        word_to_id={str(key): int(value) for key, value in payload["word_to_id"].items()},
        id_to_word={str(key): str(value) for key, value in payload["id_to_word"].items()},
        pad_token=special.get("pad", "<pad>"),
        start_token=special.get("start", "<start>"),
        end_token=special.get("end", "<end>"),
        unk_token=special.get("unk", "<unk>"),
    )


def infer_decoder_kind(keras_model) -> DecoderKind:
    layer_names = {layer.name for layer in keras_model.layers}
    if any(name.startswith("rnn_") for name in layer_names):
        return "rnn"
    if any(name.startswith("lstm_") for name in layer_names):
        return "lstm"
    raise ValueError("Could not infer decoder kind from Keras model layer names")


def build_predictor(keras_model, backend: Backend, decoder_kind: DecoderKind | None = None):
    if backend == "keras":
        return predict_with_keras_model(keras_model)

    decoder_kind = decoder_kind or infer_decoder_kind(keras_model)
    if decoder_kind == "rnn":
        return predict_with_scratch_captioner(build_scratch_rnn_captioner_from_keras(keras_model))
    if decoder_kind == "lstm":
        return predict_with_scratch_captioner(build_scratch_lstm_captioner_from_keras(keras_model))
    raise ValueError("decoder_kind must be either 'rnn' or 'lstm'")


def generate_caption(
    model_path: str | Path,
    vocabulary_path: str | Path,
    features_dir: str | Path,
    image_id: str,
    max_caption_length: int,
    backend: Backend = "keras",
    search: SearchStrategy = "greedy",
    beam_width: int = 3,
    decoder_kind: DecoderKind | None = None,
) -> dict[str, object]:
    import tensorflow as tf

    vocabulary = load_vocabulary(vocabulary_path)
    features = feature_mapping(features_dir)
    if image_id not in features:
        raise KeyError(f"Image id not found in extracted features: {image_id}")

    keras_model = tf.keras.models.load_model(model_path)
    predict_probs = build_predictor(keras_model, backend=backend, decoder_kind=decoder_kind)
    image_feature = features[image_id]

    if search == "greedy":
        token_ids = greedy_decode(predict_probs, image_feature, vocabulary, max_caption_length)
    elif search == "beam":
        token_ids = beam_search_decode(
            predict_probs,
            image_feature,
            vocabulary,
            max_caption_length,
            beam_width=beam_width,
        )
    else:
        raise ValueError("search must be either 'greedy' or 'beam'")

    return {
        "image_id": image_id,
        "caption": vocabulary.ids_to_caption(token_ids),
        "token_ids": token_ids,
        "backend": backend,
        "search": search,
        "beam_width": beam_width if search == "beam" else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate image captions with a trained RNN/LSTM decoder.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vocabulary-path", default="data/processed/captioning/vocabulary.json")
    parser.add_argument("--features-dir", default="data/features/captioning")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--max-caption-length", type=int, default=38)
    parser.add_argument("--backend", choices=("keras", "scratch"), default="keras")
    parser.add_argument("--decoder-kind", choices=("rnn", "lstm"), default=None)
    parser.add_argument("--search", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_caption(
        model_path=args.model_path,
        vocabulary_path=args.vocabulary_path,
        features_dir=args.features_dir,
        image_id=args.image_id,
        max_caption_length=args.max_caption_length,
        backend=args.backend,
        search=args.search,
        beam_width=args.beam_width,
        decoder_kind=args.decoder_kind,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
