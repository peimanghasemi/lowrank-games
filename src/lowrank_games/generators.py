"""Reproducible generators for random rank-k bimatrix games.

Every instance is defined by (family, m, n, k, seed), so benchmark sets can be
rebuilt exactly without storing the matrices.
"""
from __future__ import annotations

import numpy as np

from .game import BimatrixGame

FAMILIES = ("uniform", "gaussian", "integer", "sparse")


def low_rank_game(m: int, n: int, k: int, family: str = "uniform", seed: int = 0) -> BimatrixGame:
    """Random game with rank(A + B) = k (k = 0 gives a zero-sum game)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}, choose from {FAMILIES}")
    if not 0 <= k <= min(m, n):
        raise ValueError("k must satisfy 0 <= k <= min(m, n)")
    rng = np.random.default_rng([seed, m, n, k, FAMILIES.index(family)])

    if family == "uniform":
        A = rng.uniform(0.0, 1.0, (m, n))
        U, V = rng.uniform(-1.0, 1.0, (m, k)), rng.uniform(-1.0, 1.0, (n, k))
    elif family == "gaussian":
        A = rng.standard_normal((m, n))
        U, V = rng.standard_normal((m, k)), rng.standard_normal((n, k))
    elif family == "integer":
        A = rng.integers(0, 11, (m, n)).astype(float)
        U = rng.integers(-3, 4, (m, k)).astype(float)
        V = rng.integers(-3, 4, (n, k)).astype(float)
    else:  # sparse: most entries of A are zero, low-rank part is dense
        A = rng.uniform(0.0, 1.0, (m, n)) * (rng.random((m, n)) < 0.2)
        U, V = rng.uniform(-1.0, 1.0, (m, k)), rng.uniform(-1.0, 1.0, (n, k))

    B = U @ V.T - A
    name = f"{family}_m{m}_n{n}_k{k}_s{seed}"
    return BimatrixGame(A, B, name=name, meta=dict(family=family, m=m, n=n, k=k, seed=seed))


def benchmark_set(sizes=(10, 20, 50), ranks=(0, 1, 2, 3), families=("uniform", "integer"), seeds=range(5)):
    """Square instances used in the benchmark script."""
    for fam in families:
        for n in sizes:
            for k in ranks:
                for s in seeds:
                    yield low_rank_game(n, n, k, fam, s)
