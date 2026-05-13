from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List


def _copy_assets(src_dir: Path, dest_dir: Path, patterns: list[str]) -> int:
	dest_dir.mkdir(parents=True, exist_ok=True)
	copied = 0
	for pattern in patterns:
		for path in src_dir.rglob(pattern):
			if path.is_file():
				target = dest_dir / path.name
				shutil.copy2(path, target)
				copied += 1
	return copied

def _organize_assets_by_type(src_dir: Path, dest_root: Path) -> int:
    dest_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    
    asset_types: Dict[str, List[str]] = {
        "plots": ["*loss*.png", "*loss*.jpg", "*f1*.png", "*f1*.jpg", "*bleu*.png", "*bleu*.jpg"],
        "metrics": ["summary.csv", "summary.json", "statistics.json"],
        "predictions": ["*predictions*.json"],
        "models": ["*.keras", "*.h5"],
        "training_history": ["*history*.json"],
    }
    
    for asset_type, patterns in asset_types.items():
        type_dir = dest_root / asset_type
        type_dir.mkdir(parents=True, exist_ok=True)
        
        for pattern in patterns:
            for path in src_dir.rglob(pattern):
                if path.is_file():
                    target = type_dir / path.name
                    try:
                        shutil.copy2(path, target)
                        copied += 1
                    except Exception as e:
                        print(f"Warning: Failed to copy {path}: {e}")
    
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy and organize experiment assets into doc/figures for final report."
    )
    parser.add_argument(
        "--src-plots",
        type=Path,
        default=Path("artifacts") / "plots",
        help="Source directory containing plot assets.",
    )
    parser.add_argument(
        "--src-metrics",
        type=Path,
        default=Path("artifacts") / "reports",
        help="Source directory containing metric summaries.",
    )
    parser.add_argument(
        "--src-predictions",
        type=Path,
        default=Path("artifacts") / "predictions",
        help="Source directory containing predictions.",
    )
    parser.add_argument(
		"--dest",
		type=Path,
		default=Path("doc") / "figures",
		help="Destination directory for report assets.",
	)
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize assets into subdirectories by type.",
    )
    parser.add_argument(
		"--patterns",
		nargs="+",
		default=["*.png", "*.jpg", "*.jpeg", "*.svg", "*.json", "*.csv"],
		help="File patterns to copy.",
	)
    args = parser.parse_args()
    
    total_copied = 0
    
    if args.src_plots.exists():
        if args.organize:
            plots_dest = args.dest / "plots"
        else:
            plots_dest = args.dest
        copied = _copy_assets(args.src_plots, plots_dest, ["*.png", "*.jpg", "*.jpeg", "*.svg"])
        total_copied += copied
        if copied > 0:
            print(f"Copied {copied} plot assets to {plots_dest}")
    
    if args.src_metrics.exists():
        if args.organize:
            metrics_dest = args.dest / "metrics"
        else:
            metrics_dest = args.dest
        copied = _copy_assets(args.src_metrics, metrics_dest, ["*.json", "*.csv"])
        total_copied += copied
        if copied > 0:
            print(f"Copied {copied} metric files to {metrics_dest}")
    
    if args.src_predictions.exists() and args.organize:
        predictions_dest = args.dest / "predictions"
        copied = _copy_assets(args.src_predictions, predictions_dest, ["*.json"])
        total_copied += copied
        if copied > 0:
            print(f"Copied {copied} prediction files to {predictions_dest}")
    
    if total_copied == 0:
        print("No assets found to copy.")
    else:
        print(f"\nTotal copied: {total_copied} files -> {args.dest}")
        if args.organize:
            print(f"Assets organized by type in {args.dest}")


if __name__ == "__main__":
	main()
