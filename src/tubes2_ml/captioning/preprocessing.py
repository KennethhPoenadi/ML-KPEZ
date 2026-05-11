from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN)


@dataclass(frozen=True)
class CaptionRecord:
    image_id: str
    caption: str


@dataclass(frozen=True)
class CaptionPreprocessingConfig:
    captions_path: Path = Path("data/raw/flickr8k/captions/captions.txt")
    output_dir: Path = Path("data/processed/captioning")
    min_freq: int = 1
    max_vocab_size: int | None = None
    max_caption_length: int | None = None
    train_size: int = 6000
    validation_size: int = 1000
    test_size: int = 1000
    limit_images: int | None = None


@dataclass(frozen=True)
class CaptionPreprocessingResult:
    output_dir: Path
    vocabulary_path: Path
    metadata_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    vocab_size: int
    max_caption_length: int
    num_train_images: int
    num_validation_images: int
    num_test_images: int
    num_train_captions: int
    num_validation_captions: int
    num_test_captions: int


def clean_caption(caption: str) -> str:
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", " ", caption)
    caption = re.sub(r"\s+", " ", caption)
    return caption.strip()


def tokenize_caption(caption: str) -> list[str]:
    cleaned = clean_caption(caption)
    return cleaned.split() if cleaned else []


def image_id_from_filename(filename: str) -> str:
    return Path(filename).stem


def load_caption_records(captions_path: str | Path) -> list[CaptionRecord]:
    path = Path(captions_path)
    if not path.is_file():
        raise FileNotFoundError(f"Caption file not found: {path}")

    records: list[CaptionRecord] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or {"image", "caption"} - set(reader.fieldnames):
            raise ValueError("captions.txt must have 'image' and 'caption' columns")

        for row in reader:
            image_name = (row.get("image") or "").strip()
            caption = (row.get("caption") or "").strip()
            if image_name and caption:
                records.append(CaptionRecord(image_id=image_id_from_filename(image_name), caption=caption))

    if not records:
        raise ValueError(f"No caption records found in {path}")
    return records


def group_captions_by_image(records: Iterable[CaptionRecord]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.image_id].append(record.caption)
    return dict(grouped)


def split_image_ids(
    image_ids: Iterable[str],
    train_size: int = 6000,
    validation_size: int = 1000,
    test_size: int = 1000,
    limit_images: int | None = None,
) -> dict[str, list[str]]:
    ordered_ids = sorted(set(image_ids))
    if limit_images is not None:
        if limit_images <= 0:
            raise ValueError("limit_images must be a positive integer")
        ordered_ids = ordered_ids[:limit_images]

    if train_size <= 0 or validation_size <= 0 or test_size <= 0:
        raise ValueError("train_size, validation_size, and test_size must be positive integers")
    if len(ordered_ids) < 3:
        raise ValueError("Need at least 3 images to create train/validation/test splits")

    train_end = min(train_size, max(1, len(ordered_ids) - 2))
    validation_end = min(train_end + validation_size, len(ordered_ids) - 1)
    test_end = min(validation_end + test_size, len(ordered_ids))

    return {
        "train": ordered_ids[:train_end],
        "validation": ordered_ids[train_end:validation_end],
        "test": ordered_ids[validation_end:test_end],
    }


def build_vocabulary(
    captions: Iterable[str],
    min_freq: int = 1,
    max_vocab_size: int | None = None,
) -> tuple[dict[str, int], dict[str, str]]:
    if min_freq <= 0:
        raise ValueError("min_freq must be a positive integer")

    counter: Counter[str] = Counter()
    for caption in captions:
        counter.update(tokenize_caption(caption))

    words = [
        word
        for word, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_freq
    ]
    if max_vocab_size is not None:
        if max_vocab_size < len(SPECIAL_TOKENS):
            raise ValueError("max_vocab_size must be at least the number of special tokens")
        words = words[: max_vocab_size - len(SPECIAL_TOKENS)]

    tokens = list(SPECIAL_TOKENS) + words
    word_to_id = {token: index for index, token in enumerate(tokens)}
    id_to_word = {str(index): token for token, index in word_to_id.items()}
    return word_to_id, id_to_word


def infer_max_caption_length(captions: Iterable[str]) -> int:
    max_words = max((len(tokenize_caption(caption)) for caption in captions), default=0)
    return max_words + 1


def encode_caption_pair(
    caption: str,
    word_to_id: dict[str, int],
    max_caption_length: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if max_caption_length <= 0:
        raise ValueError("max_caption_length must be a positive integer")

    token_ids = [word_to_id.get(token, word_to_id[UNK_TOKEN]) for token in tokenize_caption(caption)]
    token_ids = token_ids[: max_caption_length - 1]
    input_ids = [word_to_id[START_TOKEN], *token_ids]
    target_ids = [*token_ids, word_to_id[END_TOKEN]]

    length = len(target_ids)

    input_array = np.full(max_caption_length, word_to_id[PAD_TOKEN], dtype=np.int32)
    target_array = np.full(max_caption_length, word_to_id[PAD_TOKEN], dtype=np.int32)
    input_array[: len(input_ids)] = input_ids
    target_array[: len(target_ids)] = target_ids
    return input_array, target_array, length


def make_split_arrays(
    image_ids: Iterable[str],
    grouped_captions: dict[str, list[str]],
    word_to_id: dict[str, int],
    max_caption_length: int,
) -> dict[str, np.ndarray]:
    sample_image_ids: list[str] = []
    input_sequences: list[np.ndarray] = []
    target_sequences: list[np.ndarray] = []
    caption_lengths: list[int] = []

    for image_id in image_ids:
        for caption in grouped_captions.get(image_id, []):
            input_ids, target_ids, length = encode_caption_pair(caption, word_to_id, max_caption_length)
            sample_image_ids.append(image_id)
            input_sequences.append(input_ids)
            target_sequences.append(target_ids)
            caption_lengths.append(length)

    return {
        "image_ids": np.asarray(sample_image_ids, dtype=str),
        "input_sequences": np.asarray(input_sequences, dtype=np.int32),
        "target_sequences": np.asarray(target_sequences, dtype=np.int32),
        "caption_lengths": np.asarray(caption_lengths, dtype=np.int32),
    }


def save_vocabulary(output_path: str | Path, word_to_id: dict[str, int], id_to_word: dict[str, str]) -> None:
    payload = {
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "special_tokens": {
            "pad": PAD_TOKEN,
            "start": START_TOKEN,
            "end": END_TOKEN,
            "unk": UNK_TOKEN,
        },
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def preprocess_captions(config: CaptionPreprocessingConfig) -> CaptionPreprocessingResult:
    records = load_caption_records(config.captions_path)
    grouped = group_captions_by_image(records)
    splits = split_image_ids(
        grouped.keys(),
        train_size=config.train_size,
        validation_size=config.validation_size,
        test_size=config.test_size,
        limit_images=config.limit_images,
    )

    train_captions = [caption for image_id in splits["train"] for caption in grouped[image_id]]
    word_to_id, id_to_word = build_vocabulary(
        train_captions,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
    )
    max_caption_length = config.max_caption_length or infer_max_caption_length(train_captions)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary_path = output_dir / "vocabulary.json"
    metadata_path = output_dir / "metadata.json"
    split_paths = {
        "train": output_dir / "train.npz",
        "validation": output_dir / "validation.npz",
        "test": output_dir / "test.npz",
    }

    save_vocabulary(vocabulary_path, word_to_id, id_to_word)

    split_arrays = {}
    for split_name, image_ids_for_split in splits.items():
        arrays = make_split_arrays(image_ids_for_split, grouped, word_to_id, max_caption_length)
        np.savez_compressed(split_paths[split_name], **arrays)
        split_arrays[split_name] = arrays

    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "vocab_size": len(word_to_id),
        "max_caption_length": max_caption_length,
        "num_images": {split_name: len(ids) for split_name, ids in splits.items()},
        "num_captions": {
            split_name: int(split_arrays[split_name]["input_sequences"].shape[0])
            for split_name in splits
        },
        "files": {
            "vocabulary": vocabulary_path.name,
            "train": split_paths["train"].name,
            "validation": split_paths["validation"].name,
            "test": split_paths["test"].name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return CaptionPreprocessingResult(
        output_dir=output_dir,
        vocabulary_path=vocabulary_path,
        metadata_path=metadata_path,
        train_path=split_paths["train"],
        validation_path=split_paths["validation"],
        test_path=split_paths["test"],
        vocab_size=len(word_to_id),
        max_caption_length=max_caption_length,
        num_train_images=len(splits["train"]),
        num_validation_images=len(splits["validation"]),
        num_test_images=len(splits["test"]),
        num_train_captions=int(split_arrays["train"]["input_sequences"].shape[0]),
        num_validation_captions=int(split_arrays["validation"]["input_sequences"].shape[0]),
        num_test_captions=int(split_arrays["test"]["input_sequences"].shape[0]),
    )


def result_to_json(result: CaptionPreprocessingResult) -> str:
    data = asdict(result)
    for key in ("output_dir", "vocabulary_path", "metadata_path", "train_path", "validation_path", "test_path"):
        data[key] = str(data[key])
    return json.dumps(data, indent=2)
