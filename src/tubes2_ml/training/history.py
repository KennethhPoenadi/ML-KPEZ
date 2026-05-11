from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def as_history_dict(history: Any) -> Dict[str, list]:
	if hasattr(history, "history"):
		history = history.history

	if not isinstance(history, dict):
		raise TypeError("history must be a dict or a Keras History-like object")

	normalized = {}
	for key, values in history.items():
		if isinstance(values, (list, tuple)):
			normalized[key] = list(values)
		else:
			normalized[key] = [values]
	return normalized


def save_history_json(path: str | Path, history: Any, extra: Dict[str, Any] | None = None) -> None:
	payload = {
		"history": as_history_dict(history),
		"extra": extra or {},
	}
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_history_json(path: str | Path) -> Dict[str, Any]:
	path = Path(path)
	return json.loads(path.read_text(encoding="utf-8"))


def merge_histories(histories: Iterable[Dict[str, list]]) -> Dict[str, list]:
	merged: Dict[str, list] = {}
	for history in histories:
		for key, values in history.items():
			merged.setdefault(key, []).extend(values)
	return merged
