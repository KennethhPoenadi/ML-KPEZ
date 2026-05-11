from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tubes2_ml.captioning.feature_extraction import (  # noqa: E402
    FeatureExtractionConfig,
    extract_flickr8k_features,
    result_to_json,
)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen CNN encoder features for Flickr8k images.")
    parser.add_argument(
        "--images-dir",
        default="data/raw/flickr8k/images",
        help="Directory containing Flickr8k image files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/features/captioning",
        help="Directory where features.npy, image_ids.json, and metadata.json are written.",
    )
    parser.add_argument(
        "--encoder",
        choices=("inception_v3", "vgg16"),
        default="inception_v3",
        help="Frozen pretrained Keras encoder to use.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Extract only the first N images for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute features when output files exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FeatureExtractionConfig(
        images_dir=resolve_project_path(args.images_dir),
        output_dir=resolve_project_path(args.output_dir),
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    result = extract_flickr8k_features(config)
    print(result_to_json(result))


if __name__ == "__main__":
    main()
