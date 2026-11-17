"""Statistics for the evaluation (success criteria R1, R2, A3)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(successes: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """Wilson (1927) score interval for a proportion. Better than the
    normal approximation for small n or proportions near 0 or 1."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def hit_at_k(ranked_sources: Sequence[str], relevant: Sequence[str], k: int) -> bool:
    """True if any of the top k results comes from a relevant source."""
    rel = set(relevant)
    return any(src in rel for src in ranked_sources[:k])


def hit_rate_sweep(results: List[Tuple[Sequence[str], Sequence[str]]], ks: Sequence[int]) -> Dict[int, dict]:
    """Hit rate at each k with its Wilson interval.
    `results` is a list of (ranked sources, relevant sources), one per question."""
    out = {}
    n = len(results)
    for k in ks:
        hits = sum(hit_at_k(r, rel, k) for r, rel in results)
        lo, hi = wilson_interval(hits, n)
        out[k] = {"hits": hits, "n": n, "rate": hits / n if n else 0.0, "low": lo, "high": hi}
    return out


def best_k(sweep: Dict[int, dict]) -> int:
    """Smallest k whose hit rate lies inside the 95% Wilson interval of the
    best k (success criterion R2): fewer pieces, same accuracy within noise."""
    best = max(sweep, key=lambda k: (sweep[k]["rate"], -k))
    lo = sweep[best]["low"]
    return min(k for k in sweep if sweep[k]["rate"] >= lo)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test (McNemar 1947) on the discordant pairs:
    b = right with retrieval only, c = right without retrieval only."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
