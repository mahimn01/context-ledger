"""Statistics for the pilot (D): a cluster bootstrap over episodes and McNemar's exact test.
Pure Python with exact Fractions; nothing here is a float.

Cluster bootstrap. A metric is a ratio of sums over episodes, sum(num) / sum(den). Each of B
resamples (10,000 by default, seeded) draws as many episodes as there are, with replacement,
and recomputes the ratio; resamples whose denominator is 0 are left out. The 95% interval is
the percentile interval by nearest rank: the ceil(0.025 B)-th and ceil(0.975 B)-th smallest of
the valid resampled values. Every metric of a run reuses the same resamples.

McNemar exact. For paired yes/no outcomes of two arms on the same questions, with b questions
right only under A and c right only under B, the two-sided p-value is
min(1, 2 * sum_{i <= min(b, c)} C(b + c, i) / 2^(b + c)), and 1 when b + c = 0.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb

RESAMPLES = 10_000
SEED = 20261031


def resample_indices(n: int, resamples: int = RESAMPLES, seed: int = SEED) -> list[list[int]]:
    """`resamples` lists of n episode indices drawn with replacement (deterministic)."""
    rng = random.Random(f"bootstrap:{seed}:{n}")
    return [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]


def nearest_rank(values: Sequence[Fraction], percent: Fraction) -> Fraction:
    """The ceil(percent/100 * n)-th smallest value (1-based, at least the first)."""
    ordered = sorted(values)
    k = Fraction(percent) * len(ordered) / 100
    rank = max(1, -(-k.numerator // k.denominator))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its 95% interval; None where undefined (no denominator)."""

    point: Fraction | None
    low: Fraction | None
    high: Fraction | None
    valid: int  # resamples with a defined value


def bootstrap(clusters: Sequence, stat: Callable[[Sequence], Fraction | None], indices: list[list[int]] | None = None) -> Estimate:
    """Point estimate `stat(clusters)` and its percentile interval over resampled clusters."""
    if not clusters:
        return Estimate(None, None, None, 0)
    indices = indices if indices is not None else resample_indices(len(clusters))
    values = [v for v in (stat([clusters[i] for i in draw]) for draw in indices) if v is not None]
    if not values:
        return Estimate(stat(clusters), None, None, 0)
    return Estimate(stat(clusters), nearest_rank(values, Fraction(5, 2)), nearest_rank(values, Fraction(195, 2)), len(values))


def ratio(clusters: Sequence[tuple[int, int]]) -> Fraction | None:
    """sum(num) / sum(den) over (num, den) clusters, or None if the denominator is 0."""
    den = sum(d for _, d in clusters)
    return Fraction(sum(n for n, _ in clusters), den) if den else None


def difference(clusters: Sequence[tuple[int, int, int, int]]) -> Fraction | None:
    """ratio of the first pair minus ratio of the second, over (n1, d1, n2, d2) clusters."""
    a = ratio([(c[0], c[1]) for c in clusters])
    b = ratio([(c[2], c[3]) for c in clusters])
    return None if a is None or b is None else a - b


def mcnemar_exact(b: int, c: int) -> Fraction:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return Fraction(1)
    tail = sum(comb(n, i) for i in range(min(b, c) + 1))
    return min(Fraction(1), Fraction(2 * tail, 2**n))


def paired_counts(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, int]:
    """(b, c, n): questions right only under A, right only under B, and how many were paired."""
    shared = sorted(set(a) & set(b))
    return sum(a[q] and not b[q] for q in shared), sum(b[q] and not a[q] for q in shared), len(shared)
