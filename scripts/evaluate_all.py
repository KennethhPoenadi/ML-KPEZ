from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tubes2_ml.experiments.metrics import (  # noqa: E402
    collect_cnn_metrics,
    collect_captioning_metrics,
    write_summary_csv,
    write_summary_json,
    generate_statistics_summary,
)
from tubes2_ml.visualization.plots import plot_history  # noqa: E402


def _load_json_metrics(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_cnn_experiments(experiments_dir: Path) -> List[Dict[str, Any]]:
    cnn_dir = experiments_dir / "cnn"
    results: List[Dict[str, Any]] = []

    if not cnn_dir.exists():
        return results

    for row in collect_cnn_metrics(cnn_dir):
        results.append({"type": "CNN", **row})

    return results


def _collect_captioning_experiments(predictions_dir: Path) -> List[Dict[str, Any]]:
    results = []
    
    if not predictions_dir.exists():
        return results
    
    for pred_file in sorted(predictions_dir.rglob("*.json")):
        try:
            data = json.loads(pred_file.read_text(encoding="utf-8"))
            
            filename_parts = pred_file.stem.split("_")
            
            backend = "keras"
            max_length = "20"
            
            if "scratch" in filename_parts:
                backend = "scratch"
            
            for part in filename_parts:
                if part.startswith("maxlen"):
                    max_length = part.replace("maxlen", "")
            
            model_name = "_".join(filename_parts)
            if backend in model_name:
                model_name = model_name.replace(f"_{backend}", "")
            if "maxlen" in model_name:
                model_name = model_name.split("_maxlen")[0]
            
            result = {
                "type": "Captioning",
                "experiment": pred_file.stem,
                "model": model_name,
                "backend": backend,
                "max_caption_length": max_length,
                "num_predictions": len(data) if isinstance(data, list) else 0,
            }
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to process {pred_file}: {e}")
    
    return results


def _collect_captioning_results_from_csv(results_csv: Path) -> List[Dict[str, Any]]:
    rows = collect_captioning_metrics(results_csv)
    return [{"type": "Captioning", **row} for row in rows]


def _plot_cnn_histories(history_dir: Path, plots_dir: Path) -> int:
    count = 0
    if not history_dir.exists():
        return count

    for metadata_path in sorted(history_dir.glob("*.json")):
        if metadata_path.name in {"summary.json", "best_model_evaluation.json"}:
            continue
        data = _load_json_metrics(metadata_path)
        if not data:
            continue
        history = data.get("history") if isinstance(data.get("history"), dict) else data
        if not isinstance(history, dict):
            continue
        plot_path = plots_dir / "cnn" / f"{metadata_path.stem}_history.png"
        plot_history(history, metrics=["loss"], title=metadata_path.stem, save_path=plot_path)
        count += 1
    return count


def _plot_captioning_histories(history_dir: Path, plots_dir: Path) -> int:
    count = 0
    if not history_dir.exists():
        return count

    for history_path in sorted(history_dir.glob("*_history.json")):
        data = _load_json_metrics(history_path)
        if not data:
            continue
        history = data.get("history")
        if not isinstance(history, dict):
            continue
        plot_path = plots_dir / "captioning" / f"{history_path.stem}.png"
        plot_history(history, title=history_path.stem, save_path=plot_path)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and collect metrics from all CNN and RNN/LSTM experiments."
    )
    parser.add_argument(
		"--experiments-dir",
		type=Path,
		default=Path("artifacts") / "experiments",
		help="Root directory that contains experiment outputs.",
    )
    parser.add_argument(
        "--captioning-results-csv",
        type=Path,
        default=Path("artifacts") / "experiments" / "captioning" / "evaluation_results.csv",
        help="CSV results from evaluate_captioning_experiments.py.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("artifacts") / "predictions",
        help="Directory containing captioning predictions.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("artifacts") / "plots",
        help="Directory for training history plots.",
    )
    parser.add_argument(
		"--out-dir",
		type=Path,
		default=Path("artifacts") / "reports",
		help="Output directory for summary files.",
	)
    parser.add_argument(
		"--format",
		choices=["csv", "json", "both"],
		default="both",
		help="Output format for summary.",
    )
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    cnn_results = _collect_cnn_experiments(args.experiments_dir)
    print(f"Collected {len(cnn_results)} CNN experiments")

    if args.captioning_results_csv.exists():
        captioning_results = _collect_captioning_results_from_csv(args.captioning_results_csv)
    else:
        captioning_results = _collect_captioning_experiments(args.predictions_dir)
    print(f"Collected {len(captioning_results)} captioning experiments")

    cnn_plot_count = _plot_cnn_histories(args.experiments_dir / "cnn", args.plots_dir)
    captioning_plot_count = _plot_captioning_histories(args.experiments_dir / "captioning", args.plots_dir)
    if cnn_plot_count or captioning_plot_count:
        print(f"Saved {cnn_plot_count + captioning_plot_count} training plots to {args.plots_dir}")
    
    all_results = cnn_results + captioning_results
    
    if args.format in ("csv", "both"):
        csv_path = args.out_dir / "evaluation_summary.csv"
        write_summary_csv(all_results, csv_path)
        print(f"Wrote evaluation summary to {csv_path}")
    
    if args.format in ("json", "both"):
        json_path = args.out_dir / "evaluation_summary.json"
        write_summary_json(all_results, json_path)
        print(f"Wrote evaluation summary to {json_path}")
        
        stats = generate_statistics_summary(all_results)
        stats_path = args.out_dir / "evaluation_statistics.json"
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"Wrote evaluation statistics to {stats_path}")
    
    if all_results:
        print(f"\nTotal experiments: {len(all_results)}")
        cnn_count = len(cnn_results)
        cap_count = len(captioning_results)
        if cnn_count > 0:
            print(f"  - CNN: {cnn_count} experiments")
        if cap_count > 0:
            print(f"  - RNN/LSTM: {cap_count} experiments")
    else:
        print("No experiments found to evaluate.")


if __name__ == "__main__":
	main()
