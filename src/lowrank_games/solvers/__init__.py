from .lemke_howson import lemke_howson
from .lp_methods import rank1_parametric, recover_from_supports, zero_sum_equilibrium
from .milp import milp_equilibrium
from .support_enumeration import first_equilibrium, support_enumeration

SOLVERS = {
    "lemke_howson": lemke_howson,
    "support_enumeration": first_equilibrium,
    "milp": milp_equilibrium,
    "rank1_parametric": rank1_parametric,
    "zero_sum_lp": zero_sum_equilibrium,
}

__all__ = ["SOLVERS", "lemke_howson", "support_enumeration", "first_equilibrium", "milp_equilibrium",
           "rank1_parametric", "zero_sum_equilibrium", "recover_from_supports"]
