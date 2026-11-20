"""Computing Nash equilibria in low-rank bimatrix games."""
from .game import BimatrixGame, Equilibrium
from .generators import benchmark_set, low_rank_game
from .solvers import SOLVERS

__version__ = "0.1.0"
__all__ = ["BimatrixGame", "Equilibrium", "low_rank_game", "benchmark_set", "SOLVERS"]
