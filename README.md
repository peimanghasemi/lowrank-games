# lowrank-games

Computing Nash equilibria in low-rank bimatrix games.

A bimatrix game (A, B) has rank k if rank(A + B) = k. Rank 0 is a zero-sum game and can be solved by linear programming, while general bimatrix games are PPAD-hard. Low-rank games sit in between, and the low-rank structure can be used to design faster methods. This repository contains exact solvers, a generator for random rank-k instances and a benchmark that compares the methods.

## Methods

| Solver | Idea | Applies to |
|---|---|---|
| `lemke_howson` | complementary pivoting with a lexicographic ratio test (works for degenerate games) | any game |
| `support_enumeration` | enumerates equal-size supports and solves the indifference conditions | small nondegenerate games |
| `milp` | MILP of Sandholm, Gilpin and Conitzer (2005), solved with HiGHS; can also select the welfare-maximizing equilibrium | any game, small to medium size |
| `zero_sum_lp` | minimax LP | rank 0 |
| `rank1_parametric` | parametric LP that uses A + B = u v^T (see below) | rank 0 and 1 |

Every equilibrium returned by any solver is checked: both players' regrets must be below a tolerance.

### Rank-1 parametric method

Write A + B = u v^T and fix lambda = u^T x. The column player's payoff becomes x^T B y = lambda v^T y - x^T A y, so for fixed lambda the problem is the zero-sum LP

```
max  z
s.t. z <= (A^T x)_j - lambda v_j    for all j      (duals: y_j)
     1^T x = 1,  u^T x = lambda,  x >= 0             (dual of the last one: mu)
```

A saddle point of this LP is a Nash equilibrium of (A, B) if and only if mu(lambda) = 0. Since lambda only appears in the right-hand side, the dual feasible set does not depend on lambda and mu(lambda) is piecewise constant. The solver brackets a sign change of mu on [min u, max u], bisects until the two ends are neighbouring pieces, computes the breakpoint where both dual solutions are optimal, and mixes them so that mu = 0. Together with any optimal x at the breakpoint this is an exact equilibrium. Every step only solves LPs of the original size.

## Instances

`low_rank_game(m, n, k, family, seed)` draws A at random and sets B = U V^T - A with U (m x k) and V (n x k), so that rank(A + B) = k exactly. Families: `uniform`, `gaussian`, `integer` (small integer payoffs, often degenerate) and `sparse`. An instance is fully defined by its parameters, so benchmark sets can be rebuilt exactly. `scripts/make_instances.py` exports them as `.npz` files with a manifest for use in other codes.

## Usage

```bash
pip install -e ".[dev]"
pytest
```

```python
from lowrank_games import low_rank_game
from lowrank_games.solvers import lemke_howson, rank1_parametric

game = low_rank_game(50, 50, k=1, family="uniform", seed=0)
eq = rank1_parametric(game)
print(game.is_equilibrium(eq.x, eq.y), eq.info)
```

Benchmark:

```bash
python scripts/run_benchmark.py                      # n = 10, 20, 50, 100; k = 0..3; 2 families; 5 seeds
python scripts/run_benchmark.py --milp-max-n 20      # also run the MILP on n = 20
```

Results go to `results/results.csv` and `results/summary.md`.

## Results

Mean time in seconds over 40 square instances per size (ranks 0 to 3, uniform and integer families, 5 seeds), one x86-64 CPU core. All 400 instances were solved by every applicable method. The rank-1 method is reported for rank 0 and 1 only.

| n | Lemke-Howson | rank-1 parametric | MILP (HiGHS) | support enumeration |
|---|---|---|---|---|
| 10 | 0.0007 | 0.072 | 0.26 | 1.31 |
| 20 | 0.0017 | 0.091 | - | - |
| 50 | 0.0129 | 0.144 | - | - |
| 100 | 0.131 | 0.406 | - | - |

Lemke-Howson is very fast on random instances but has exponential worst cases, while the rank-1 method only solves a sequence of LPs. The MILP becomes slow from about n = 20 (several instances hit a 30 s limit), but it can optimize over the set of equilibria, which the other methods cannot.

## Adding a method

A solver is a function `solver(game: BimatrixGame) -> Equilibrium`. Add it to `SOLVERS` in `src/lowrank_games/solvers/__init__.py` and to `applicable()` in `scripts/run_benchmark.py`; the tests and the benchmark then pick it up.

## References

- R. Kannan and T. Theobald (2010). Games of fixed rank: a hierarchy of bimatrix games. Economic Theory 42, 157-173.
- B. Adsul, J. Garg, R. Mehta and M. Sohoni (2011). Rank-1 bimatrix games: a homeomorphism and a polynomial time algorithm. STOC.
- C. E. Lemke and J. T. Howson (1964). Equilibrium points of bimatrix games. SIAM Journal on Applied Mathematics 12, 413-423.
- T. Sandholm, A. Gilpin and V. Conitzer (2005). Mixed-integer programming methods for finding Nash equilibria. AAAI.

## License

MIT
