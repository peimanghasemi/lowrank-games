"""Support enumeration for nondegenerate games (exponential, for small instances)."""
from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np

from ..game import BimatrixGame, Equilibrium


def _indifferent(M: np.ndarray, I: tuple, J: tuple) -> np.ndarray | None:
    """Mix over J that makes every strategy in I indifferent under payoff M (rows I)."""
    s = len(J)
    lhs = np.zeros((len(I) + 1, s + 1))
    lhs[:-1, :s] = M[np.ix_(I, J)]
    lhs[:-1, s] = -1.0
    lhs[-1, :s] = 1.0
    rhs = np.zeros(len(I) + 1)
    rhs[-1] = 1.0
    try:
        sol = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    return sol[:s]


def support_enumeration(game: BimatrixGame, tol: float = 1e-9) -> Iterator[Equilibrium]:
    m, n = game.shape
    for size in range(1, min(m, n) + 1):
        for I in combinations(range(m), size):
            for J in combinations(range(n), size):
                yJ = _indifferent(game.A, I, J)            # makes row player indifferent on I
                xI = _indifferent(game.B.T, J, I)          # makes column player indifferent on J
                if yJ is None or xI is None or yJ.min() < -tol or xI.min() < -tol:
                    continue
                x = np.zeros(m); x[list(I)] = xI
                y = np.zeros(n); y[list(J)] = yJ
                if game.is_equilibrium(x, y):
                    yield Equilibrium(x, y, "support_enumeration", {"support_size": size})


def first_equilibrium(game: BimatrixGame) -> Equilibrium:
    return next(support_enumeration(game))
