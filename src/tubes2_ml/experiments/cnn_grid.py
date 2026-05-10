from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from tubes2_ml.cnn.models import SharedConvCNNConfig
from tubes2_ml.cnn.train import CNNTrainingConfig, train_shared_conv_cnn


DEFAULT_LAYER_VARIANTS = ((32,), (32, 64))
DEFAULT_FILTER_VARIANTS = ((16, 32), (32, 64))
DEFAULT_KERNEL_VARIANTS = ((3,), (5,))
DEFAULT_POOLING_VARIANTS = ("max", "average")


def make_experiment_name(
    conv_filters: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    pooling_type: str,
) -> str:
    filters = "f" + "-".join(str(value) for value in conv_filters)
    kernels = "k" + "-".join(str(value) for value in kernel_sizes)
    return f"cnn_{len(conv_filters)}conv_{filters}_{kernels}_{pooling_type}pool"


def generate_shared_conv_grid(
    base_config: SharedConvCNNConfig | None = None,
    layer_variants: tuple[tuple[int, ...], ...] = DEFAULT_LAYER_VARIANTS,
    filter_variants: tuple[tuple[int, ...], ...] = DEFAULT_FILTER_VARIANTS,
    kernel_variants: tuple[tuple[int, ...], ...] = DEFAULT_KERNEL_VARIANTS,
    pooling_variants: tuple[str, ...] = DEFAULT_POOLING_VARIANTS,
) -> list[SharedConvCNNConfig]:
    """Generate the required 16 shared Conv2D experiment configurations."""
    base = base_config or SharedConvCNNConfig()
    configs: list[SharedConvCNNConfig] = []

    for layer_template, filter_template, kernel_template, pooling_type in product(
        layer_variants,
        filter_variants,
        kernel_variants,
        pooling_variants,
    ):
        num_layers = len(layer_template)
        conv_filters = _resize_tuple(filter_template, num_layers)
        kernel_sizes = _resize_tuple(kernel_template, num_layers)
        name = make_experiment_name(conv_filters, kernel_sizes, pooling_type)
        configs.append(
            replace(
                base,
                conv_filters=conv_filters,
                kernel_sizes=kernel_sizes,
                pooling_type=pooling_type,
                name=name,
            )
        )

    return configs


def _resize_tuple(values: tuple[int, ...], length: int) -> tuple[int, ...]:
    if len(values) >= length:
        return values[:length]
    return values + (values[-1],) * (length - len(values))


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML experiment configs.") from exc

    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return data


def run_shared_conv_grid(
    model_config: SharedConvCNNConfig,
    training_config: CNNTrainingConfig,
) -> list[dict[str, Any]]:
    results = []
    for experiment_config in generate_shared_conv_grid(model_config):
        _, _, metadata = train_shared_conv_cnn(experiment_config, training_config)
        results.append(metadata)
    return results


def _tuple_from_config(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def config_from_yaml(path: str | Path) -> tuple[SharedConvCNNConfig, CNNTrainingConfig]:
    data = load_yaml_config(path)
    model_data = data.get("model", {})
    training_data = data.get("training", {})

    model_config = SharedConvCNNConfig(
        input_shape=tuple(model_data.get("input_shape", (150, 150, 3))),
        num_classes=int(model_data.get("num_classes", 6)),
        conv_filters=_tuple_from_config(model_data.get("conv_filters", (32, 64))),
        kernel_sizes=_tuple_from_config(model_data.get("kernel_sizes", (3, 3))),
        pooling_type=str(model_data.get("pooling_type", "max")),
        dense_units=_tuple_from_config(model_data.get("dense_units", (128,))),
        dropout_rate=float(model_data.get("dropout_rate", 0.3)),
        learning_rate=float(model_data.get("learning_rate", 1e-3)),
        activation=str(model_data.get("activation", "relu")),
        name=str(model_data.get("name", "shared_conv_cnn")),
    )
    training_config = CNNTrainingConfig(
        train_dir=training_data.get("train_dir", "data/raw/intel_image_classification/seg_train/seg_train"),
        validation_dir=training_data.get("validation_dir"),
        output_dir=training_data.get("output_dir", "models/keras/cnn"),
        history_dir=training_data.get("history_dir", "artifacts/experiments/cnn"),
        image_size=tuple(training_data.get("image_size", (150, 150))),
        batch_size=int(training_data.get("batch_size", 32)),
        epochs=int(training_data.get("epochs", 10)),
        validation_split=float(training_data.get("validation_split", 0.2)),
        seed=int(training_data.get("seed", 42)),
    )
    return model_config, training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 16 shared Conv2D CNN experiments.")
    parser.add_argument("--config", default="configs/cnn/shared_conv.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print generated configs without training.")
    args = parser.parse_args()

    model_config, training_config = config_from_yaml(args.config)
    grid = generate_shared_conv_grid(model_config)

    if args.dry_run:
        for config in grid:
            print(config)
        print(f"Total experiments: {len(grid)}")
        return

    run_shared_conv_grid(model_config, training_config)


if __name__ == "__main__":
    main()
