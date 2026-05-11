from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tubes2_ml.captioning.preprocessing import (  # noqa: E402
    CaptionPreprocessingConfig,
    preprocess_captions,
    result_to_json,
)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Flickr8k captions for RNN/LSTM decoder training.")
    parser.add_argument(
        "--captions-path",
        default="data/raw/flickr8k/captions/captions.txt",
        help="CSV file with image and caption columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/captioning",
        help="Directory where vocabulary.json, metadata.json, and split npz files are written.",
    )
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--max-caption-length", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--limit-images", type=int, default=None, help="Use only the first N image ids for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CaptionPreprocessingConfig(
        captions_path=resolve_project_path(args.captions_path),
        output_dir=resolve_project_path(args.output_dir),
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        max_caption_length=args.max_caption_length,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        limit_images=args.limit_images,
    )
    result = preprocess_captions(config)
    print(result_to_json(result))


if __name__ == "__main__":
    main()
