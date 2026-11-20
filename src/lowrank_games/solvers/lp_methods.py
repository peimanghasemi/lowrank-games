"""LP-based methods: zero-sum games (rank 0) and a parametric method for rank-1 games.

Rank-1 idea (in the spirit of Kannan and Theobald, and Adsul et al.):
write A + B = u v^T and fix lambda = u^T x. Then the column player's payoff
x^T B y equals lambda v^T y - x^T A y, so for fixed lambda the game becomes the
zero-sum LP

    max_x,z  z   s.t.  z <= (A^T x)_j - lambda v_j  for all j,
                       1^T x = 1,  u^T x = lambda,  x >= 0.

Its inequality duals give the column player's strategy y. A saddle point is a
Nash equilibrium of (A, B) exactly when the multiplier mu(lambda) of the
constraint u^T x = lambda is zero.

lambda only enters the right-hand side, so the dual feasible set does not
depend on lambda and mu(lambda) is piecewise constant. The method brackets a
sign change of mu, bisects until the two ends belong to neighbouring pieces,
computes the breakpoint lambda_c where both dual solutions are optimal, and
mixes the two duals so that mu = 0. With any optimal x at lambda_c this gives
an exact equilibrium.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ..game import BimatrixGame, Equilibrium


def zero_sum_equilibrium(game: BimatrixGame) -> Equilibrium:
    """Minimax LP for B = -A (works for any game with rank(A + B) = 0)."""
    A = game.A
    m, n = A.shape
    c = np.r_[np.zeros(m), -1.0]
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=np.r_[np.ones(m), 0.0][None],
                  b_eq=[1.0], bounds=[(0, None)] * m + [(None, None)], method="highs")
    x = res.x[:m]
    y = -res.ineqlin.marginals
    return Equilibrium(x / x.sum(), y / y.sum(), "zero_sum_lp", {"value": -res.fun})


def _parametric_lp(A: np.ndarray, u: np.ndarray, v: np.ndarray, lam: float):
    m, n = A.shape
    c = np.r_[np.zeros(m), -1.0]
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    A_eq = np.vstack([np.r_[np.ones(m), 0.0], np.r_[u, 0.0]])
    res = linprog(c, A_ub=A_ub, b_ub=-lam * v, A_eq=A_eq, b_eq=[1.0, lam],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    if res.status != 0:
        return None
    y = np.clip(-res.ineqlin.marginals, 0, None)
    mu = float(res.eqlin.marginals[1])
    # dual objective as a line in lambda: D(lambda) = c0 + lambda * slope
    c0, slope = float(res.eqlin.marginals[0]), float(v @ y + mu)
    return res.x[:m], y, mu, c0, slope


def recover_from_supports(game: BimatrixGame, I: np.ndarray, J: np.ndarray) -> Equilibrium | None:
    """Exact equilibrium with supp(x) in I and supp(y) in J, if one exists (two LPs)."""
    m, n = game.shape

    def mix(M: np.ndarray, S: np.ndarray, T: np.ndarray, k: int):
        # strategy p on T (length k) such that rows in S of M p are maximal: M_S p = w, M_rest p <= w
        others = np.setdiff1d(np.arange(M.shape[0]), S)
        A_eq = np.vstack([np.hstack([M[np.ix_(S, T)], -np.ones((len(S), 1))]), np.r_[np.ones(len(T)), 0.0]])
        b_eq = np.r_[np.zeros(len(S)), 1.0]
        A_ub = np.hstack([M[np.ix_(others, T)], -np.ones((len(others), 1))]) if len(others) else None
        b_ub = np.zeros(len(others)) if len(others) else None
        res = linprog(np.zeros(len(T) + 1), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0, None)] * len(T) + [(None, None)], method="highs")
        if res.status != 0:
            return None
        p = np.zeros(k); p[T] = res.x[: len(T)]
        return p

    y = mix(game.A, I, J, n)
    x = mix(game.B.T, J, I, m)
    if x is None or y is None or not game.is_equilibrium(x, y):
        return None
    return Equilibrium(x, y, "recovered")


def rank1_parametric(game: BimatrixGame, grid: int = 32, max_refinements: int = 5,
                     bisection_steps: int = 60) -> Equilibrium:
    if game.rank() > 1:
        raise ValueError("rank1_parametric needs rank(A + B) <= 1")
    if game.rank() == 0:
        eq = zero_sum_equilibrium(game)
        eq.method = "rank1_parametric"
        return eq
    U, V = game.factor(1)
    u, v = U[:, 0], V[:, 0]
    A = game.A
    evals = 0

    def solve(lam: float):
        nonlocal evals
        evals += 1
        return _parametric_lp(A, u, v, lam)

    def done(x, y, lam, how):
        if game.is_equilibrium(x / x.sum(), y / y.sum()):
            return Equilibrium(x / x.sum(), y / y.sum(), "rank1_parametric",
                               {"lam": float(lam), "lp_solves": evals, "step": how})
        return None

    for level in range(max_refinements):
        lams = np.linspace(u.min(), u.max(), grid * 2 ** level + 1)
        sols = [solve(l) for l in lams]
        for lam, s in zip(lams, sols):
            if s is not None and (eq := done(s[0], s[1], lam, "grid")):
                return eq
        for a, b, sa, sb in zip(lams[:-1], lams[1:], sols[:-1], sols[1:]):
            if sa is None or sb is None or np.sign(sa[2]) == np.sign(sb[2]):
                continue
            for _ in range(bisection_steps):          # shrink to neighbouring pieces
                mid = 0.5 * (a + b)
                sm = solve(mid)
                if sm is None:
                    break
                if (eq := done(sm[0], sm[1], mid, "bisection")):
                    return eq
                if np.isclose(sm[2], sa[2]):
                    a, sa = mid, sm
                elif np.isclose(sm[2], sb[2]):
                    b, sb = mid, sm
                elif np.sign(sm[2]) == np.sign(sa[2]):
                    a, sa = mid, sm
                else:
                    b, sb = mid, sm
            # breakpoint where both duals are optimal, then mix the duals so that mu = 0
            if abs(sa[4] - sb[4]) < 1e-14:
                continue
            lam_c = (sb[3] - sa[3]) / (sa[4] - sb[4])
            sc = solve(lam_c)
            if sc is None:
                continue
            t = sb[2] / (sb[2] - sa[2])
            y = t * sa[1] + (1 - t) * sb[1]
            if (eq := done(sc[0], y, lam_c, "breakpoint")):
                return eq
            I, J = np.flatnonzero(sc[0] > 1e-9), np.flatnonzero(y > 1e-9)
            if (eq := recover_from_supports(game, I, J)) is not None:
                eq.method = "rank1_parametric"
                eq.info.update(lam=float(lam_c), lp_solves=evals, step="recovered")
                return eq
    raise RuntimeError("rank1_parametric: no equilibrium found")
