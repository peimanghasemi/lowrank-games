"""Lemke-Howson algorithm with a lexicographic ratio test (safe for degenerate games).

Labels 0..m-1 belong to the row player's pure strategies and m..m+n-1 to the
column player's. Tableau T1 encodes A y <= 1 (variables y, slacks r) and
tableau T2 encodes B^T x <= 1 (variables x, slacks s).
"""
from __future__ import annotations

import numpy as np

from ..game import BimatrixGame, Equilibrium


class _Tableau:
    def __init__(self, M: np.ndarray, var_labels: list[int], slack_labels: list[int]):
        rows = M.shape[0]
        # columns: one per label (variables then slacks), then the right-hand side
        self.labels = var_labels + slack_labels
        self.col = {lab: j for j, lab in enumerate(self.labels)}
        self.T = np.hstack([M, np.eye(rows), np.ones((rows, 1))])
        self.basis = list(slack_labels)
        self.slack_cols = [self.col[l] for l in slack_labels]

    def pivot(self, entering: int) -> int:
        """Bring `entering` into the basis and return the label that leaves."""
        c = self.col[entering]
        column = self.T[:, c]
        rows = np.flatnonzero(column > 1e-12)
        if rows.size == 0:
            raise RuntimeError("unbounded ray, the payoff matrices must be positive")
        # lexicographic minimum ratio: (rhs, slack columns) divided by pivot entry
        keys = np.column_stack([self.T[rows, -1], self.T[np.ix_(rows, self.slack_cols)]]) / column[rows, None]
        order = np.lexsort(keys.T[::-1])
        r = rows[order[0]]
        self.T[r] /= self.T[r, c]
        others = np.arange(len(self.T)) != r
        self.T[others] -= np.outer(self.T[others, c], self.T[r])
        leaving, self.basis[r] = self.basis[r], entering
        return leaving

    def values(self, labels: range) -> np.ndarray:
        v = np.zeros(len(labels))
        for r, lab in enumerate(self.basis):
            if lab in labels:
                v[labels.index(lab)] = self.T[r, -1]
        return v


def lemke_howson(game: BimatrixGame, dropped_label: int = 0, max_pivots: int = 100_000) -> Equilibrium:
    m, n = game.shape
    # shift payoffs so that all entries are positive (equilibria are unchanged)
    A = game.A - game.A.min() + 1.0
    B = game.B - game.B.min() + 1.0
    rows, cols = range(m), range(m, m + n)

    t1 = _Tableau(A, var_labels=list(cols), slack_labels=list(rows))      # A y + r = 1
    t2 = _Tableau(B.T, var_labels=list(rows), slack_labels=list(cols))    # B^T x + s = 1

    entering = dropped_label
    tab = t2 if dropped_label in rows else t1
    for it in range(max_pivots):
        leaving = tab.pivot(entering)
        if leaving == dropped_label:
            break
        entering = leaving
        tab = t1 if tab is t2 else t2
    else:
        raise RuntimeError("Lemke-Howson did not terminate")

    x, y = t2.values(rows), t1.values(cols)
    return Equilibrium(x / x.sum(), y / y.sum(), "lemke_howson", {"pivots": it + 1, "dropped_label": dropped_label})
