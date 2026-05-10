from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Model does not contain Conv2D layers")


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image: np.ndarray,
    layer_name: str | None = None,
    class_index: int | None = None,
) -> np.ndarray:
    layer_name = layer_name or find_last_conv_layer_name(model)
    image_batch = _as_single_image_batch(image)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if float(max_value.numpy()) == 0.0:
        return np.zeros(heatmap.shape, dtype=np.float32)

    return (heatmap / max_value).numpy().astype(np.float32)


def overlay_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.35,
    cmap: str = "jet",
) -> np.ndarray:
    image = _to_float_image(image)
    heatmap = np.asarray(heatmap, dtype=np.float32)

    heatmap_rgb = plt.get_cmap(cmap)(heatmap)[..., :3]
    heatmap_rgb = tf.image.resize(
        heatmap_rgb,
        size=(image.shape[0], image.shape[1]),
        method="bilinear",
    ).numpy()

    overlay = (1 - alpha) * image + alpha * heatmap_rgb
    return np.clip(overlay, 0.0, 1.0)


def plot_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.35,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    overlay = overlay_gradcam(image, heatmap, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(_to_float_image(image))
    axes[0].set_title("Image")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig


def save_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    output_path: str | Path,
    layer_name: str | None = None,
    class_index: int | None = None,
    alpha: float = 0.35,
):
    heatmap = make_gradcam_heatmap(
        model=model,
        image=image,
        layer_name=layer_name,
        class_index=class_index,
    )
    fig = plot_gradcam(image, heatmap, alpha=alpha, save_path=output_path)
    plt.close(fig)
    return heatmap


def _as_single_image_batch(image: np.ndarray) -> tf.Tensor:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[np.newaxis, ...]
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects one image with shape (H, W, C) or (1, H, W, C)")
    return tf.convert_to_tensor(image)


def _to_float_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 4:
        image = image[0]
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)
