from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tubes2_ml.visualization.plots import plot_history


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_runs(experiments_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(experiments_dir.glob("*.json")):
        if path.name in {"best_model_evaluation.json", "summary.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "model_config" not in payload or "history" not in payload:
            continue
        payload["_path"] = str(path)
        payload["_name"] = path.stem
        runs.append(payload)
    if not runs:
        raise FileNotFoundError(f"No CNN experiment metadata found in {experiments_dir}")
    return runs


def metric(run: dict[str, Any]) -> float:
    return float(run.get("metrics", {}).get("validation_macro_f1", 0.0))


def conv_layers(run: dict[str, Any]) -> str:
    return str(len(run["model_config"]["conv_filters"]))


def filter_combo(run: dict[str, Any]) -> str:
    return "-".join(str(value) for value in run["model_config"]["conv_filters"])


def kernel_combo(run: dict[str, Any]) -> str:
    return "-".join(str(value) for value in run["model_config"]["kernel_sizes"])


def pooling(run: dict[str, Any]) -> str:
    return str(run["model_config"]["pooling_type"])


def group_summary(runs: list[dict[str, Any]], name: str, key_fn: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        grouped[key_fn(run)].append(metric(run))

    rows = []
    for value, scores in sorted(grouped.items()):
        rows.append(
            {
                "factor": name,
                "value": value,
                "mean_validation_macro_f1": mean(scores),
                "best_validation_macro_f1": max(scores),
                "num_runs": len(scores),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_histories(runs: list[dict[str, Any]], plots_dir: Path) -> list[str]:
    paths: list[str] = []
    for run in runs:
        history = run.get("history", {})
        if not history:
            continue
        path = plots_dir / f"{run['_name']}_loss.png"
        plot_history(history, metrics=["loss"], title=run["_name"], save_path=path)
        paths.append(str(path))
    return paths


def best_by_factor(rows: list[dict[str, Any]], factor: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["factor"] == factor]
    return max(candidates, key=lambda row: row["mean_validation_macro_f1"])


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_report(
    runs: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    evaluation: dict[str, Any] | None,
    plot_paths: list[str],
) -> str:
    best = max(runs, key=metric)
    layer_best = best_by_factor(summary_rows, "conv_layers")
    filter_best = best_by_factor(summary_rows, "filters")
    kernel_best = best_by_factor(summary_rows, "kernels")
    pooling_best = best_by_factor(summary_rows, "pooling")

    lines = [
        "# CNN Analysis Report",
        "",
        "## Experiment Summary",
        "",
        f"- Total eksperimen: {len(runs)}",
        f"- Model terbaik berdasarkan validation macro F1: `{best['_name']}`",
        f"- Validation macro F1 terbaik: `{metric(best):.6f}`",
        f"- Plot loss tersimpan: `{len(plot_paths)}` file di `artifacts/plots/cnn/history`",
        "",
        "## Hyperparameter Findings",
        "",
        f"- Jumlah layer conv terbaik secara rata-rata: `{layer_best['value']}` layer, mean macro F1 `{layer_best['mean_validation_macro_f1']:.6f}`.",
        f"- Kombinasi filter terbaik secara rata-rata: `{filter_best['value']}`, mean macro F1 `{filter_best['mean_validation_macro_f1']:.6f}`.",
        f"- Kombinasi kernel terbaik secara rata-rata: `{kernel_best['value']}`, mean macro F1 `{kernel_best['mean_validation_macro_f1']:.6f}`.",
        f"- Pooling terbaik secara rata-rata: `{pooling_best['value']}`, mean macro F1 `{pooling_best['mean_validation_macro_f1']:.6f}`.",
        "",
        "## Evaluation Findings",
        "",
    ]

    if evaluation is None:
        lines.extend(
            [
                "- Evaluasi test split Keras vs scratch belum dijalankan.",
                "- Jalankan `python3 scripts/evaluate_cnn_best.py` untuk menghasilkan `artifacts/experiments/cnn/best_model_evaluation.json`.",
            ]
        )
    else:
        metrics = evaluation["metrics"]
        params = evaluation["parameter_counts"]
        lines.extend(
            [
                f"- Model evaluasi: `{evaluation['best_experiment']}`",
                f"- Keras test macro F1: `{metrics['keras']['macro_f1']:.6f}` pada `{metrics['keras']['num_samples']}` sampel.",
                f"- Scratch shared macro F1: `{metrics['scratch_shared']['macro_f1']:.6f}` pada `{metrics['scratch_shared']['num_samples']}` sampel.",
                f"- Scratch non-shared macro F1: `{metrics['scratch_non_shared']['macro_f1']:.6f}` pada `{metrics['scratch_non_shared']['num_samples']}` sampel.",
                f"- Agreement scratch shared vs Keras pada prefix yang sama: `{metrics['scratch_shared'].get('prediction_agreement_with_keras_same_prefix', 0.0):.6f}`.",
                f"- Agreement scratch non-shared vs Keras pada prefix yang sama: `{metrics['scratch_non_shared'].get('prediction_agreement_with_keras_same_prefix', 0.0):.6f}`.",
                f"- Parameter scratch shared: `{params['scratch_shared']}`.",
                f"- Parameter scratch non-shared: `{params['scratch_non_shared']}`.",
                f"- Rasio parameter non-shared/shared: `{evaluation['parameter_ratio_non_shared_vs_shared']:.2f}`.",
                "",
                "Scratch non-shared di evaluasi ini mengganti Conv2D dengan LocallyConnected2D memakai bobot Conv2D yang direplikasi per posisi. Karena bobotnya direplikasi, prediksi bisa sangat dekat dengan shared, tetapi jumlah parameter jauh lebih besar. Ini menunjukkan parameter sharing lebih efisien untuk pola visual yang berulang di berbagai lokasi gambar.",
            ]
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Artifact training yang ada memakai `epochs=5` dan `early_stopping_patience=1`, sedangkan `configs/cnn/shared_conv.yaml` saat ini menulis `epochs=20`. Samakan atau jelaskan di laporan.",
            "- Macro F1 test scratch bisa mahal karena forward propagation NumPy menjalankan sliding window eksplisit.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_cnn_report(
    experiments_dir: Path,
    plots_dir: Path,
    reports_dir: Path,
    evaluation_json: Path,
) -> dict[str, Any]:
    runs = load_runs(experiments_dir)
    plot_paths = plot_histories(runs, plots_dir)
    summary_rows = (
        group_summary(runs, "conv_layers", conv_layers)
        + group_summary(runs, "filters", filter_combo)
        + group_summary(runs, "kernels", kernel_combo)
        + group_summary(runs, "pooling", pooling)
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = reports_dir / "cnn_hparam_summary.csv"
    summary_json = reports_dir / "cnn_hparam_summary.json"
    report_md = reports_dir / "cnn_analysis.md"

    write_csv(summary_rows, summary_csv)
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    evaluation = load_optional_json(evaluation_json)
    report_md.write_text(make_report(runs, summary_rows, evaluation, plot_paths), encoding="utf-8")

    return {
        "num_runs": len(runs),
        "num_plots": len(plot_paths),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "report_md": str(report_md),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CNN plots and analysis report from experiment metadata.")
    parser.add_argument("--experiments-dir", default="artifacts/experiments/cnn")
    parser.add_argument("--plots-dir", default="artifacts/plots/cnn/history")
    parser.add_argument("--reports-dir", default="artifacts/reports")
    parser.add_argument("--evaluation-json", default="artifacts/experiments/cnn/best_model_evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_cnn_report(
        experiments_dir=resolve_project_path(args.experiments_dir),
        plots_dir=resolve_project_path(args.plots_dir),
        reports_dir=resolve_project_path(args.reports_dir),
        evaluation_json=resolve_project_path(args.evaluation_json),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
