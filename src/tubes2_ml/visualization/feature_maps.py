from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_conv_layer_names(model: tf.keras.Model) -> list[str]:
    return [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]


def extract_feature_maps(
    model: tf.keras.Model,
    images: np.ndarray,
    layer_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    if layer_names is None:
        layer_names = get_conv_layer_names(model)

    if not layer_names:
        raise ValueError("Model does not contain Conv2D layers")

    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(name).output for name in layer_names],
    )
    outputs = feature_model.predict(_as_batch(images), verbose=0)

    if len(layer_names) == 1:
        outputs = [outputs]

    return {name: np.asarray(output) for name, output in zip(layer_names, outputs)}


def plot_feature_maps(
    feature_maps: np.ndarray,
    max_channels: int = 16,
    title: str | None = None,
    cmap: str = "viridis",
    save_path: str | Path | None = None,
):
    maps = np.asarray(feature_maps)
    if maps.ndim == 4:
        maps = maps[0]
    if maps.ndim != 3:
        raise ValueError("feature_maps must have shape (H, W, C) or (N, H, W, C)")

    channels = min(max_channels, maps.shape[-1])
    cols = min(4, channels)
    rows = int(np.ceil(channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for channel_idx in range(rows * cols):
        ax = axes[channel_idx // cols, channel_idx % cols]
        ax.axis("off")
        if channel_idx < channels:
            ax.imshow(maps[..., channel_idx], cmap=cmap)
            ax.set_title(f"Channel {channel_idx}", fontsize=9)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig


def save_conv_feature_visualizations(
    model: tf.keras.Model,
    images: np.ndarray,
    output_dir: str | Path,
    layer_names: list[str] | None = None,
    max_channels: int = 16,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    feature_maps_by_layer = extract_feature_maps(model, images, layer_names=layer_names)
    saved_paths = {}

    for layer_name, feature_maps in feature_maps_by_layer.items():
        figure_path = output_path / f"{layer_name}_feature_maps.png"
        fig = plot_feature_maps(
            feature_maps,
            max_channels=max_channels,
            title=f"{layer_name} feature maps",
            save_path=figure_path,
        )
        plt.close(fig)
        saved_paths[layer_name] = figure_path

    return saved_paths


def _as_batch(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    if images.ndim != 4:
        raise ValueError("images must have shape (H, W, C) or (N, H, W, C)")
    return images
