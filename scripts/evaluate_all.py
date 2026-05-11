from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_metrics(path: Path) -> Dict[str, float]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	if isinstance(payload, dict) and "metrics" in payload and isinstance(payload["metrics"], dict):
		payload = payload["metrics"]

	metrics: Dict[str, float] = {}
	if isinstance(payload, dict):
		for key, value in payload.items():
			if isinstance(value, (int, float)):
				metrics[key] = float(value)
	return metrics


def _collect_metrics(experiments_dir: Path) -> List[Dict[str, float]]:
	rows: List[Dict[str, float]] = []
	for metrics_path in experiments_dir.rglob("metrics.json"):
		metrics = _load_metrics(metrics_path)
		experiment_name = metrics_path.parent.name
		row = {"experiment": experiment_name, **metrics}
		rows.append(row)
	return rows


def _write_summary(rows: List[Dict[str, float]], out_dir: Path) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	summary_path = out_dir / "summary.json"
	summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

	if not rows:
		return

	keys = ["experiment"] + sorted({k for row in rows for k in row.keys() if k != "experiment"})
	lines = [",".join(keys)]
	for row in rows:
		values = [str(row.get(key, "")) for key in keys]
		lines.append(",".join(values))

	(out_dir / "summary.csv").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description="Collect experiment metrics into a single summary.")
	parser.add_argument(
		"--experiments-dir",
		type=Path,
		default=Path("artifacts") / "experiments",
		help="Root directory that contains experiment outputs.",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=Path("artifacts") / "reports",
		help="Output directory for summary files.",
	)
	args = parser.parse_args()

	rows = _collect_metrics(args.experiments_dir)
	_write_summary(rows, args.out_dir)

	if not rows:
		print("No metrics.json found.")
		return

	print(f"Collected {len(rows)} experiment summaries -> {args.out_dir}")


if __name__ == "__main__":
	main()
