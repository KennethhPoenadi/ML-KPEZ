from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List
import csv


@dataclass(frozen=True)
class MetricsPayload:
    experiment: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] | None = None


def write_metrics_json(path: str | Path, payload: MetricsPayload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(payload)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def load_metrics_json(path: str | Path) -> MetricsPayload:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = {key: float(value) for key, value in data.get("metrics", {}).items()}
    return MetricsPayload(
        experiment=str(data.get("experiment", path.parent.name)),
        metrics=metrics,
        metadata=data.get("metadata"),
    )


def ensure_experiment_dir(base_dir: str | Path, experiment: str) -> Path:
    run_dir = Path(base_dir) / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def collect_cnn_metrics(history_dir: str | Path) -> List[Dict[str, Any]]:
    history_dir = Path(history_dir)
    results = []
    
    if not history_dir.exists():
        return results
    
    for metadata_path in sorted(history_dir.glob("*.json")):
        if metadata_path.name in {"summary.json", "best_model_evaluation.json"}:
            continue
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            config = data.get("model_config", {})
            metrics = data.get("metrics", {})
            result = {
                "experiment": metadata_path.stem,
                "model_name": config.get("name", ""),
                "conv_filters": str(config.get("conv_filters", "")),
                "kernel_sizes": str(config.get("kernel_sizes", "")),
                "pooling_type": config.get("pooling_type", ""),
                "dropout_rate": config.get("dropout_rate", 0),
                "validation_macro_f1": metrics.get("validation_macro_f1", 0.0),
                "test_macro_f1": metrics.get("test_macro_f1", 0.0),
            }
            result.update({k: v for k, v in metrics.items() if k not in result})
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load metrics from {metadata_path}: {e}")
    
    return results


def collect_captioning_metrics(results_csv: str | Path) -> List[Dict[str, Any]]:
    results_csv = Path(results_csv)
    results = []
    
    if not results_csv.exists():
        return results
    
    try:
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized: Dict[str, Any] = {}
                for key, value in row.items():
                    if value is None:
                        normalized[key] = value
                        continue
                    if value == "":
                        normalized[key] = value
                        continue
                    try:
                        normalized[key] = float(value)
                    except ValueError:
                        normalized[key] = value
                results.append(normalized)
    except Exception as e:
        print(f"Warning: Failed to load captioning metrics from {results_csv}: {e}")
    
    return results


def write_summary_csv(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not rows:
        return
    
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    
    sorted_keys = []
    for key in ["experiment", "model", "model_name", "decoder_type", "backend"]:
        if key in all_keys:
            sorted_keys.append(key)
            all_keys.discard(key)
    sorted_keys.extend(sorted(all_keys))
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in sorted_keys})


def write_summary_json(rows: List[Dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def generate_statistics_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    
    summary = {
        "total_experiments": len(rows),
        "metric_ranges": {},
    }
    
    numeric_fields = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                numeric_fields.add(key)
            elif isinstance(value, str):
                try:
                    float(value)
                    numeric_fields.add(key)
                except (ValueError, TypeError):
                    pass
    
    for field in numeric_fields:
        values = []
        for row in rows:
            val = row.get(field)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass
        
        if values:
            summary["metric_ranges"][field] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values),
            }
    
    return summary
