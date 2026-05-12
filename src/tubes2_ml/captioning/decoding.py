from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

PredictProbsFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CaptionVocabulary:
    word_to_id: dict[str, int]
    id_to_word: dict[str, str]
    pad_token: str = "<pad>"
    start_token: str = "<start>"
    end_token: str = "<end>"
    unk_token: str = "<unk>"

    @property
    def pad_id(self) -> int:
        return self.word_to_id[self.pad_token]

    @property
    def start_id(self) -> int:
        return self.word_to_id[self.start_token]

    @property
    def end_id(self) -> int:
        return self.word_to_id[self.end_token]

    def ids_to_words(self, token_ids: list[int] | np.ndarray, skip_special: bool = True) -> list[str]:
        special = {self.pad_token, self.start_token, self.end_token}
        words: list[str] = []
        for token_id in token_ids:
            word = self.id_to_word.get(str(int(token_id)), self.unk_token)
            if word == self.end_token:
                break
            if skip_special and word in special:
                continue
            words.append(word)
        return words

    def ids_to_caption(self, token_ids: list[int] | np.ndarray) -> str:
        return " ".join(self.ids_to_words(token_ids))


def make_padded_input(prefix_ids: list[int], max_caption_length: int, pad_id: int) -> np.ndarray:
    if max_caption_length <= 0:
        raise ValueError("max_caption_length must be positive")
    sequence = np.full((1, max_caption_length), pad_id, dtype=np.int32)
    clipped = prefix_ids[:max_caption_length]
    sequence[0, : len(clipped)] = clipped
    return sequence


def make_padded_batch(prefixes: list[list[int]], max_caption_length: int, pad_id: int) -> np.ndarray:
    if max_caption_length <= 0:
        raise ValueError("max_caption_length must be positive")
    sequences = np.full((len(prefixes), max_caption_length), pad_id, dtype=np.int32)
    for index, prefix_ids in enumerate(prefixes):
        clipped = prefix_ids[:max_caption_length]
        sequences[index, : len(clipped)] = clipped
    return sequences


def greedy_decode(
    predict_probs: PredictProbsFn,
    image_feature: np.ndarray,
    vocabulary: CaptionVocabulary,
    max_caption_length: int,
) -> list[int]:
    prefix = [vocabulary.start_id]
    generated: list[int] = []

    for position in range(max_caption_length):
        model_input = make_padded_input(prefix, max_caption_length, vocabulary.pad_id)
        probabilities = predict_probs(np.asarray(image_feature, dtype=np.float32)[np.newaxis, :], model_input)
        next_id = int(np.argmax(probabilities[0, position]))
        if next_id == vocabulary.end_id:
            break
        generated.append(next_id)
        prefix.append(next_id)

    return generated


def greedy_decode_batch(
    predict_probs: PredictProbsFn,
    image_features: np.ndarray,
    vocabulary: CaptionVocabulary,
    max_caption_length: int,
) -> list[list[int]]:
    features = np.asarray(image_features, dtype=np.float32)
    if features.ndim == 1:
        features = features[np.newaxis, :]
    if features.ndim != 2:
        raise ValueError("image_features must have shape (batch, feature_dim)")

    batch_size = features.shape[0]
    prefixes = [[vocabulary.start_id] for _ in range(batch_size)]
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    active = np.ones(batch_size, dtype=bool)

    for position in range(max_caption_length):
        if not np.any(active):
            break

        model_input = make_padded_batch(prefixes, max_caption_length, vocabulary.pad_id)
        probabilities = predict_probs(features, model_input)
        next_ids = np.argmax(probabilities[:, position, :], axis=-1).astype(np.int64)

        for index, next_id in enumerate(next_ids):
            if not active[index]:
                continue
            token_id = int(next_id)
            if token_id == vocabulary.end_id:
                active[index] = False
                continue
            generated[index].append(token_id)
            prefixes[index].append(token_id)

    return generated


def beam_search_decode(
    predict_probs: PredictProbsFn,
    image_feature: np.ndarray,
    vocabulary: CaptionVocabulary,
    max_caption_length: int,
    beam_width: int = 3,
) -> list[int]:
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    beams: list[tuple[list[int], float, bool]] = [([vocabulary.start_id], 0.0, False)]
    feature_batch = np.asarray(image_feature, dtype=np.float32)[np.newaxis, :]

    for position in range(max_caption_length):
        candidates: list[tuple[list[int], float, bool]] = []
        for prefix, score, ended in beams:
            if ended:
                candidates.append((prefix, score, True))
                continue

            model_input = make_padded_input(prefix, max_caption_length, vocabulary.pad_id)
            probabilities = predict_probs(feature_batch, model_input)[0, position]
            top_ids = np.argsort(probabilities)[-beam_width:][::-1]
            for token_id in top_ids:
                token_id = int(token_id)
                probability = max(float(probabilities[token_id]), 1e-12)
                next_prefix = prefix + [token_id]
                candidates.append((next_prefix, score + math.log(probability), token_id == vocabulary.end_id))

        beams = sorted(candidates, key=lambda item: item[1] / max(1, len(item[0]) - 1), reverse=True)[:beam_width]
        if all(ended for _, _, ended in beams):
            break

    best_prefix = beams[0][0]
    generated = best_prefix[1:]
    if vocabulary.end_id in generated:
        generated = generated[: generated.index(vocabulary.end_id)]
    return generated


def predict_with_keras_model(keras_model) -> PredictProbsFn:
    def predict_probs(image_features: np.ndarray, caption_input_ids: np.ndarray) -> np.ndarray:
        return np.asarray(keras_model.predict([image_features, caption_input_ids], verbose=0), dtype=np.float32)

    return predict_probs


def predict_with_scratch_captioner(captioner) -> PredictProbsFn:
    def predict_probs(image_features: np.ndarray, caption_input_ids: np.ndarray) -> np.ndarray:
        return np.asarray(captioner.forward(image_features, caption_input_ids), dtype=np.float32)

    return predict_probs
