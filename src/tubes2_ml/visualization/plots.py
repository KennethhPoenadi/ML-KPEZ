from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def plot_history(
	history: Dict[str, List[float]],
	metrics: Iterable[str] | None = None,
	title: str | None = None,
	save_path: str | Path | None = None,
	show: bool = False,
) -> None:
	if metrics is None:
		metrics = [k for k in history.keys() if not k.startswith("val_")]

	for metric in metrics:
		values = history.get(metric)
		if values is None:
			continue
		plt.plot(values, label=metric)
		val_key = f"val_{metric}"
		if val_key in history:
			plt.plot(history[val_key], label=val_key)

	plt.xlabel("epoch")
	plt.ylabel("value")
	if title:
		plt.title(title)
	plt.legend()
	plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, bbox_inches="tight")

	if show:
		plt.show()
	plt.close()


def plot_metric(
	values: List[float],
	title: str,
	ylabel: str,
	save_path: str | Path | None = None,
	show: bool = False,
) -> None:
	plt.plot(values)
	plt.xlabel("epoch")
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, bbox_inches="tight")

	if show:
		plt.show()
	plt.close()
