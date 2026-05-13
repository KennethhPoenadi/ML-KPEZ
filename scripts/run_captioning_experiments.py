from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tubes2_ml.captioning.models import CaptionDecoderConfig, DecoderType, InjectionMode  # noqa: E402
from tubes2_ml.captioning.train import (  # noqa: E402
    CaptionTrainingConfig,
    architecture_version,
    experiment_name,
    make_base_model_config,
    train_caption_decoder,
)

DEFAULT_DECODERS: tuple[DecoderType, ...] = ("rnn", "lstm")
DEFAULT_LAYER_VARIANTS = (1, 2, 3)
DEFAULT_HIDDEN_VARIANTS = (128, 512)


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        return {}

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to read captioning config files. Run `pip install pyyaml`.") from exc

    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def int_tuple(values: list[int] | tuple[int, ...] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    return tuple(int(value) for value in values)


def decoder_tuple(values: list[str] | tuple[str, ...] | None) -> tuple[DecoderType, ...]:
    if values is None:
        return DEFAULT_DECODERS
    decoders = tuple(str(value).lower() for value in values)
    invalid = [value for value in decoders if value not in {"rnn", "lstm"}]
    if invalid:
        raise ValueError(f"Unsupported decoder types: {invalid}")
    return decoders  # type: ignore[return-value]


def injection_tuple(values: list[str] | tuple[str, ...] | None) -> tuple[InjectionMode, ...]:
    if values is None:
        return ("pre",)
    modes = tuple(str(value).lower() for value in values)
    invalid = [value for value in modes if value not in {"pre", "init"}]
    if invalid:
        raise ValueError(f"Unsupported injection modes: {invalid}")
    return modes  # type: ignore[return-value]


def configs_from_yaml(path: str | Path) -> tuple[CaptionDecoderConfig, CaptionTrainingConfig, dict[str, tuple]]:
    data = load_yaml_config(path)
    model_data = data.get("model", {})
    training_data = data.get("training", {})
    grid_data = data.get("grid", {})

    base_model_config = CaptionDecoderConfig(
        vocab_size=1,
        feature_dim=1,
        max_caption_length=1,
        embed_dim=int(model_data.get("embed_dim", 256)),
        hidden_units=int(model_data.get("hidden_units", 256)),
        num_recurrent_layers=int(model_data.get("num_recurrent_layers", 1)),
        dropout_rate=float(model_data.get("dropout_rate", 0.0)),
        learning_rate=float(model_data.get("learning_rate", 1e-3)),
        decoder_type=str(model_data.get("decoder_type", "lstm")).lower(),  # type: ignore[arg-type]
        injection_mode=str(model_data.get("injection_mode", "pre")).lower(),  # type: ignore[arg-type]
        name=str(model_data.get("name", "caption_decoder")),
    )
    training_config = CaptionTrainingConfig(
        processed_dir=resolve_project_path(training_data.get("processed_dir", "data/processed/captioning")),
        features_dir=resolve_project_path(training_data.get("features_dir", "data/features/captioning")),
        output_dir=resolve_project_path(training_data.get("output_dir", "models/keras/captioning")),
        history_dir=resolve_project_path(training_data.get("history_dir", "artifacts/experiments/captioning")),
        batch_size=int(training_data.get("batch_size", 64)),
        epochs=int(training_data.get("epochs", 10)),
        seed=int(training_data.get("seed", 42)),
    )
    grid = {
        "decoders": decoder_tuple(grid_data.get("decoders")),
        "injection_modes": injection_tuple(grid_data.get("injection_modes")),
        "num_recurrent_layers": int_tuple(grid_data.get("num_recurrent_layers"), DEFAULT_LAYER_VARIANTS),
        "hidden_units": int_tuple(grid_data.get("hidden_units"), DEFAULT_HIDDEN_VARIANTS),
    }
    return base_model_config, training_config, grid


def generate_experiment_configs(
    base_config: CaptionDecoderConfig,
    decoders: tuple[DecoderType, ...] = DEFAULT_DECODERS,
    injection_modes: tuple[InjectionMode, ...] = ("pre",),
    num_recurrent_layers: tuple[int, ...] = DEFAULT_LAYER_VARIANTS,
    hidden_units: tuple[int, ...] = DEFAULT_HIDDEN_VARIANTS,
) -> list[CaptionDecoderConfig]:
    configs: list[CaptionDecoderConfig] = []
    for decoder_type, injection_mode, layer_count, hidden_size in product(
        decoders,
        injection_modes,
        num_recurrent_layers,
        hidden_units,
    ):
        base = make_base_model_config(decoder_type, layer_count, hidden_size, injection_mode=injection_mode)
        configs.append(
            replace(
                base,
                embed_dim=base_config.embed_dim,
                dropout_rate=base_config.dropout_rate,
                learning_rate=base_config.learning_rate,
            )
        )
    return configs


def _jsonable_dataclass(value: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(value), default=str))


def _metadata_matches(
    metadata_path: Path,
    model_config: CaptionDecoderConfig,
    training_config: CaptionTrainingConfig,
) -> bool:
    if not metadata_path.exists():
        return False

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_model_config = dict(metadata.get("model_config", {}))
    expected_model_config = _jsonable_dataclass(model_config)
    metadata_training_config = dict(metadata.get("training_config", {}))
    expected_training_config = _jsonable_dataclass(training_config)

    for ignored_key in ("vocab_size", "feature_dim", "max_caption_length", "name"):
        metadata_model_config.pop(ignored_key, None)
        expected_model_config.pop(ignored_key, None)

    expected_architecture_version = architecture_version(model_config.injection_mode)
    metadata_architecture_version = metadata.get("architecture_version")
    version_matches = metadata_architecture_version == expected_architecture_version or (
        metadata_architecture_version is None and model_config.injection_mode == "pre"
    )

    return (
        metadata_model_config == expected_model_config
        and metadata_training_config == expected_training_config
        and version_matches
    )


def run_grid(
    model_configs: list[CaptionDecoderConfig],
    training_config: CaptionTrainingConfig,
    skip_completed: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    history_dir = Path(training_config.history_dir)

    for model_config in model_configs:
        metadata_path = history_dir / f"{experiment_name(model_config)}_metadata.json"
        if skip_completed and _metadata_matches(metadata_path, model_config, training_config):
            print(f"Skipping completed experiment: {model_config.name}")
            continue
        if metadata_path.exists():
            print(f"Re-running experiment with updated config: {model_config.name}")

        result = train_caption_decoder(model_config, training_config)
        results.append(asdict(result))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RNN/LSTM captioning experiments.")
    parser.add_argument("--config", default="configs/captioning/hparam_grid.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the 12 experiment configs without training.")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Train experiments even when matching metadata already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model_config, training_config, grid = configs_from_yaml(args.config)
    experiment_configs = generate_experiment_configs(base_model_config, **grid)

    if args.dry_run:
        payload = {
            "training_config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(training_config).items()
            },
            "experiments": [asdict(config) for config in experiment_configs],
            "total_experiments": len(experiment_configs),
        }
        print(json.dumps(payload, indent=2))
        return

    results = run_grid(experiment_configs, training_config, skip_completed=not args.rerun_completed)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
