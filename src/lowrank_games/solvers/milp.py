"""Mixed-integer formulation of Sandholm, Gilpin and Conitzer (2005), solved with HiGHS.

Binary s_i = 1 means pure strategy i is outside the support; then its regret may be
positive but its probability must be zero. The objective can select among equilibria,
for example the one with maximum social welfare.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..game import BimatrixGame, Equilibrium


def milp_equilibrium(game: BimatrixGame, objective: str = "feasibility", time_limit: float = 60.0) -> Equilibrium:
    A, B = game.A, game.B
    m, n = game.shape
    UA, UB = A.max() - A.min(), B.max() - B.min()

    # variable layout: x(m) y(n) u1 u2 r1(m) r2(n) s1(m) s2(n)
    ix = np.arange(m); iy = m + np.arange(n); iu1, iu2 = m + n, m + n + 1
    ir1 = m + n + 2 + np.arange(m); ir2 = ir1[-1] + 1 + np.arange(n)
    is1 = ir2[-1] + 1 + np.arange(m); is2 = is1[-1] + 1 + np.arange(n)
    N = is2[-1] + 1

    rows, lo, hi = [], [], []

    def add(coefs: dict, l: float, h: float) -> None:
        row = np.zeros(N)
        for idx, val in coefs.items():
            row[idx] += val
        rows.append(row); lo.append(l); hi.append(h)

    add({**{i: 1.0 for i in ix}}, 1, 1)
    add({**{j: 1.0 for j in iy}}, 1, 1)
    for i in range(m):   # u1 - (A y)_i - r1_i = 0
        add({**{iy[j]: -A[i, j] for j in range(n)}, iu1: 1.0, ir1[i]: -1.0}, 0, 0)
        add({ix[i]: 1.0, is1[i]: 1.0}, -np.inf, 1)            # x_i <= 1 - s1_i
        add({ir1[i]: 1.0, is1[i]: -UA}, -np.inf, 0)           # r1_i <= UA s1_i
    for j in range(n):   # u2 - (B^T x)_j - r2_j = 0
        add({**{ix[i]: -B[i, j] for i in range(m)}, iu2: 1.0, ir2[j]: -1.0}, 0, 0)
        add({iy[j]: 1.0, is2[j]: 1.0}, -np.inf, 1)
        add({ir2[j]: 1.0, is2[j]: -UB}, -np.inf, 0)

    c = np.zeros(N)
    if objective == "welfare":
        c[iu1] = c[iu2] = -1.0
    elif objective != "feasibility":
        raise ValueError("objective must be 'feasibility' or 'welfare'")

    lb = np.zeros(N); ub = np.full(N, np.inf)
    lb[[iu1, iu2]] = -np.inf
    ub[np.r_[ix, iy]] = 1.0
    ub[np.r_[is1, is2]] = 1.0
    integrality = np.zeros(N); integrality[np.r_[is1, is2]] = 1

    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality,
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        raise RuntimeError(f"MILP failed: {res.message}")
    x, y = np.clip(res.x[ix], 0, None), np.clip(res.x[iy], 0, None)
    return Equilibrium(x / x.sum(), y / y.sum(), f"milp_{objective}", {"status": res.status, "mip_gap": getattr(res, "mip_gap", None)})
