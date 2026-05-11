from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
	return _TOKEN_RE.findall(text.lower())


def _ensure_tokens(sentence: Sequence[str] | str) -> List[str]:
	if isinstance(sentence, str):
		return _tokenize(sentence)
	return [str(tok).lower() for tok in sentence]


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
	return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(
	references: Iterable[Sequence[str] | str],
	hypothesis: Sequence[str] | str,
	max_n: int = 4,
	smooth: float = 1.0,
) -> float:
	refs = [_ensure_tokens(ref) for ref in references]
	hyp = _ensure_tokens(hypothesis)

	if not refs or not hyp:
		return 0.0

	precisions = []
	for n in range(1, max_n + 1):
		hyp_counts = _ngram_counts(hyp, n)
		max_ref_counts: Counter = Counter()
		for ref in refs:
			ref_counts = _ngram_counts(ref, n)
			for k, v in ref_counts.items():
				if v > max_ref_counts[k]:
					max_ref_counts[k] = v

		overlap = 0
		for k, v in hyp_counts.items():
			overlap += min(v, max_ref_counts.get(k, 0))

		denom = max(1, sum(hyp_counts.values()))
		precisions.append((overlap + smooth) / (denom + smooth))

	log_precision = sum(math.log(p) for p in precisions) / max_n

	ref_lens = [len(ref) for ref in refs]
	hyp_len = len(hyp)
	closest_ref_len = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))
	if hyp_len == 0:
		brevity_penalty = 0.0
	elif hyp_len > closest_ref_len:
		brevity_penalty = 1.0
	else:
		brevity_penalty = math.exp(1.0 - (closest_ref_len / hyp_len))

	return float(brevity_penalty * math.exp(log_precision))


def corpus_bleu(
	references_list: Iterable[Iterable[Sequence[str] | str]],
	hypotheses: Iterable[Sequence[str] | str],
	max_n: int = 4,
	smooth: float = 1.0,
) -> float:
	total_score = 0.0
	count = 0
	for refs, hyp in zip(references_list, hypotheses):
		total_score += sentence_bleu(refs, hyp, max_n=max_n, smooth=smooth)
		count += 1
	return float(total_score / count) if count else 0.0


def _chunk_count(matches: List[int]) -> int:
	if not matches:
		return 0
	chunks = 1
	for i in range(1, len(matches)):
		if matches[i] != matches[i - 1] + 1:
			chunks += 1
	return chunks


def meteor_score(
	references: Iterable[Sequence[str] | str],
	hypothesis: Sequence[str] | str,
	alpha: float = 0.9,
	beta: float = 3.0,
	gamma: float = 0.5,
) -> float:
	refs = [_ensure_tokens(ref) for ref in references]
	hyp = _ensure_tokens(hypothesis)

	if not refs or not hyp:
		return 0.0

	best_score = 0.0
	for ref in refs:
		ref_index = {tok: [] for tok in ref}
		for i, tok in enumerate(ref):
			ref_index.setdefault(tok, []).append(i)

		matches = []
		used = set()
		for tok in hyp:
			positions = ref_index.get(tok, [])
			pos = next((p for p in positions if p not in used), None)
			if pos is not None:
				used.add(pos)
				matches.append(pos)

		if not matches:
			continue

		matches.sort()
		m = len(matches)
		precision = m / len(hyp)
		recall = m / len(ref)
		f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

		chunks = _chunk_count(matches)
		penalty = gamma * ((chunks / m) ** beta)
		score = (1 - penalty) * f_mean
		best_score = max(best_score, score)

	return float(best_score)
