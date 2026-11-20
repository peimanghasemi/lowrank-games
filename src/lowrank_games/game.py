"""Bimatrix games, low-rank structure and equilibrium checks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BimatrixGame:
    """Two-player game: the row player gets x^T A y, the column player x^T B y."""

    A: np.ndarray
    B: np.ndarray
    name: str = "game"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        if self.A.shape != self.B.shape or self.A.ndim != 2:
            raise ValueError("A and B must be matrices of the same shape")

    @property
    def shape(self) -> tuple[int, int]:
        return self.A.shape

    @property
    def C(self) -> np.ndarray:
        """Sum of payoffs A + B. The game has rank k if rank(C) = k."""
        return self.A + self.B

    def rank(self, tol: float = 1e-9) -> int:
        s = np.linalg.svd(self.C, compute_uv=False)
        return int((s > tol * max(1.0, s[0] if s.size else 0.0)).sum())

    def factor(self, k: int | None = None, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
        """Return U (m x k) and V (n x k) with A + B = U V^T, from a truncated SVD."""
        Uf, s, Vt = np.linalg.svd(self.C, full_matrices=False)
        k = self.rank(tol) if k is None else k
        root = np.sqrt(s[:k])
        return Uf[:, :k] * root, Vt[:k].T * root

    def payoffs(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        return float(x @ self.A @ y), float(x @ self.B @ y)

    def regrets(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Gain each player could get by deviating to a pure best response."""
        Ay, Btx = self.A @ y, self.B.T @ x
        return float(Ay.max() - x @ Ay), float(Btx.max() - x @ self.B @ y)

    def is_equilibrium(self, x: np.ndarray, y: np.ndarray, tol: float = 1e-7) -> bool:
        if not (_is_mixed(x, len(self.A)) and _is_mixed(y, self.A.shape[1])):
            return False
        r1, r2 = self.regrets(x, y)
        scale = max(1.0, np.abs(self.A).max(), np.abs(self.B).max())
        return max(r1, r2) <= tol * scale


def _is_mixed(p: np.ndarray, dim: int, tol: float = 1e-8) -> bool:
    p = np.asarray(p, dtype=float)
    return p.shape == (dim,) and p.min() >= -tol and abs(p.sum() - 1.0) <= 1e-6


@dataclass
class Equilibrium:
    x: np.ndarray
    y: np.ndarray
    method: str
    info: dict = field(default_factory=dict)

    def clean(self, tol: float = 1e-12) -> "Equilibrium":
        x = np.where(self.x < tol, 0.0, self.x)
        y = np.where(self.y < tol, 0.0, self.y)
        return Equilibrium(x / x.sum(), y / y.sum(), self.method, self.info)
