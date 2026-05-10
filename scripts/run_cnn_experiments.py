from __future__ import annotations
import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any
from tubes2_ml.cnn.models import SharedConvCNNConfig
from tubes2_ml.cnn.train import CNNTrainingConfig, train_shared_conv_cnn

LAYER_VARIANTS = ((32,), (32, 64))
FILTER_VARIANTS = ((16, 32), (32, 64))
KERNEL_VARIANTS = ((3,), (5,))
POOLING_VARIANTS = ("max", "average")

def make_experiment_name(
    conv_filters: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    pooling_type: str,
) -> str:
    filters = "f" + "-".join(str(value) for value in conv_filters)
    kernels = "k" + "-".join(str(value) for value in kernel_sizes)
    return f"cnn_{len(conv_filters)}conv_{filters}_{kernels}_{pooling_type}pool"

def resize_tuple(values: tuple[int, ...], length: int) -> tuple[int, ...]:
    if len(values) >= length:
        return values[:length]
    return values + (values[-1],) * (length - len(values))

def generate_shared_conv_grid(base_config: SharedConvCNNConfig) -> list[SharedConvCNNConfig]:
    configs = []

    for layer_template, filter_template, kernel_template, pooling_type in product(
        LAYER_VARIANTS,
        FILTER_VARIANTS,
        KERNEL_VARIANTS,
        POOLING_VARIANTS,
    ):
        num_layers = len(layer_template)
        conv_filters = resize_tuple(filter_template, num_layers)
        kernel_sizes = resize_tuple(kernel_template, num_layers)

        configs.append(
            replace(
                base_config,
                conv_filters=conv_filters,
                kernel_sizes=kernel_sizes,
                pooling_type=pooling_type,
                name=make_experiment_name(conv_filters, kernel_sizes, pooling_type),
            )
        )

    return configs

def load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def tuple_from_config(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)

def config_from_yaml(path: str | Path) -> tuple[SharedConvCNNConfig, CNNTrainingConfig]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[2]
    data = load_yaml_config(config_path)
    model_data = data.get("model", {})
    training_data = data.get("training", {})

    model_config = SharedConvCNNConfig(
        input_shape=tuple(model_data.get("input_shape", (150, 150, 3))),
        num_classes=int(model_data.get("num_classes", 6)),
        conv_filters=tuple_from_config(model_data.get("conv_filters", (32, 64))),
        kernel_sizes=tuple_from_config(model_data.get("kernel_sizes", (3, 3))),
        pooling_type=str(model_data.get("pooling_type", "max")),
        dense_units=tuple_from_config(model_data.get("dense_units", (128,))),
        dropout_rate=float(model_data.get("dropout_rate", 0.3)),
        learning_rate=float(model_data.get("learning_rate", 1e-3)),
        activation=str(model_data.get("activation", "relu")),
        name=str(model_data.get("name", "shared_conv_cnn")),
    )
    training_config = CNNTrainingConfig(
        train_dir=resolve_project_path(
            project_root,
            training_data.get("train_dir", "data/raw/intel_image_classification/seg_train/seg_train"),
        ),
        validation_dir=resolve_optional_project_path(project_root, training_data.get("validation_dir")),
        output_dir=resolve_project_path(project_root, training_data.get("output_dir", "models/keras/cnn")),
        history_dir=resolve_project_path(project_root, training_data.get("history_dir", "artifacts/experiments/cnn")),
        image_size=tuple(training_data.get("image_size", (150, 150))),
        batch_size=int(training_data.get("batch_size", 32)),
        epochs=int(training_data.get("epochs", 10)),
        validation_split=float(training_data.get("validation_split", 0.2)),
        seed=int(training_data.get("seed", 42)),
    )
    return model_config, training_config

def resolve_project_path(project_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path

def resolve_optional_project_path(project_root: Path, path: str | Path | None) -> Path | None:
    if path in {None, ""}:
        return None
    return resolve_project_path(project_root, path)

def run_grid(model_config: SharedConvCNNConfig, training_config: CNNTrainingConfig) -> None:
    for experiment_config in generate_shared_conv_grid(model_config):
        train_shared_conv_cnn(experiment_config, training_config)

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

    run_grid(model_config, training_config)


if __name__ == "__main__":
    main()
