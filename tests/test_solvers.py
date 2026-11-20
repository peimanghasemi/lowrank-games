import numpy as np
import pytest

from lowrank_games import BimatrixGame, low_rank_game
from lowrank_games.solvers import (lemke_howson, milp_equilibrium, rank1_parametric,
                                   support_enumeration, zero_sum_equilibrium)

FAMILIES = ["uniform", "gaussian", "integer", "sparse"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_generator_rank(family, k):
    g = low_rank_game(12, 9, k, family, seed=3)
    assert g.rank() == k
    U, V = g.factor()
    assert np.allclose(U @ V.T, g.C)


def test_generator_is_reproducible():
    g1, g2 = low_rank_game(10, 10, 2, "uniform", 7), low_rank_game(10, 10, 2, "uniform", 7)
    assert np.array_equal(g1.A, g2.A) and np.array_equal(g1.B, g2.B)


def test_matching_pennies():
    A = np.array([[1, -1], [-1, 1]])
    g = BimatrixGame(A, -A)
    for eq in (lemke_howson(g), zero_sum_equilibrium(g), milp_equilibrium(g), rank1_parametric(g)):
        assert np.allclose(eq.x, [0.5, 0.5]) and np.allclose(eq.y, [0.5, 0.5])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", range(5))
def test_lemke_howson_all_labels(family, seed):
    g = low_rank_game(7, 6, 2, family, seed)
    for label in range(13):
        eq = lemke_howson(g, dropped_label=label)
        assert g.is_equilibrium(eq.x, eq.y)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("seed", range(5))
def test_support_enumeration_and_milp(family, seed):
    g = low_rank_game(5, 5, 1, family, seed)
    eqs = list(support_enumeration(g))
    assert eqs and all(g.is_equilibrium(e.x, e.y) for e in eqs)
    eq = milp_equilibrium(g, objective="welfare")
    assert g.is_equilibrium(eq.x, eq.y)
    # the welfare-maximizing equilibrium is at least as good as every enumerated one
    best = max(sum(g.payoffs(e.x, e.y)) for e in eqs)
    assert sum(g.payoffs(eq.x, eq.y)) >= best - 1e-6


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n", [3, 8, 25])
@pytest.mark.parametrize("seed", range(5))
def test_rank1_parametric(family, n, seed):
    g = low_rank_game(n, n + 2, 1, family, seed)
    eq = rank1_parametric(g)
    assert g.is_equilibrium(eq.x, eq.y)


def test_rank1_rejects_higher_rank():
    with pytest.raises(ValueError):
        rank1_parametric(low_rank_game(5, 5, 2))
