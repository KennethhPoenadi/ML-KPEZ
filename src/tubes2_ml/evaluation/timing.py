from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator


@dataclass
class Timer:
	durations: Dict[str, float] = field(default_factory=dict)

	@contextmanager
	def track(self, name: str) -> Generator[None, None, None]:
		start = time.perf_counter()
		try:
			yield
		finally:
			self.durations[name] = time.perf_counter() - start


def time_call(func, *args, **kwargs):
	start = time.perf_counter()
	result = func(*args, **kwargs)
	duration = time.perf_counter() - start
	return result, duration


@contextmanager
def time_block(name: str, records: Dict[str, float]) -> Generator[None, None, None]:
	start = time.perf_counter()
	try:
		yield
	finally:
		records[name] = time.perf_counter() - start
