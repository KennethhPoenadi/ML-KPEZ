from __future__ import annotations

import argparse
import shutil
from pathlib import Path


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


def main() -> None:
	parser = argparse.ArgumentParser(description="Copy plots and report assets into doc/figures.")
	parser.add_argument(
		"--src",
		type=Path,
		default=Path("artifacts") / "plots",
		help="Source directory containing plot assets.",
	)
	parser.add_argument(
		"--dest",
		type=Path,
		default=Path("doc") / "figures",
		help="Destination directory for report assets.",
	)
	parser.add_argument(
		"--patterns",
		nargs="+",
		default=["*.png", "*.jpg", "*.jpeg", "*.svg"],
		help="File patterns to copy.",
	)
	args = parser.parse_args()

	copied = _copy_assets(args.src, args.dest, args.patterns)
	if copied == 0:
		print("No assets found to copy.")
	else:
		print(f"Copied {copied} assets -> {args.dest}")


if __name__ == "__main__":
	main()
