"""Run all applicable solvers on the benchmark set and write results/results.csv and results/summary.md.

Usage:
    python scripts/run_benchmark.py                         # default set
    python scripts/run_benchmark.py --sizes 10 20 --seeds 3  # smaller run
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from lowrank_games import SOLVERS, benchmark_set

# which solvers to try for an instance of size n and rank k
def applicable(name: str, n: int, k: int, milp_max_n: int) -> bool:
    if name == "support_enumeration":
        return n <= 10
    if name == "zero_sum_lp":
        return k == 0
    if name == "rank1_parametric":
        return k <= 1
    if name == "milp":
        return n <= milp_max_n
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50, 100])
    p.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--families", nargs="+", default=["uniform", "integer"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--milp-time", type=float, default=30.0)
    p.add_argument("--milp-max-n", type=int, default=10)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    rows = []
    for game in benchmark_set(a.sizes, a.ranks, a.families, range(a.seeds)):
        n, k = game.meta["n"], game.meta["k"]
        for name, solver in SOLVERS.items():
            if not applicable(name, n, k, a.milp_max_n):
                continue
            t0 = time.perf_counter()
            try:
                eq = solver(game, time_limit=a.milp_time) if name == "milp" else solver(game)
                ok = game.is_equilibrium(eq.x, eq.y)
                regret = max(game.regrets(eq.x, eq.y))
                support = f"{int((eq.x > 1e-9).sum())}x{int((eq.y > 1e-9).sum())}"
            except Exception:
                ok, regret, support = False, float("nan"), ""
            rows.append(dict(instance=game.name, **game.meta, solver=name, solved=ok,
                             seconds=round(time.perf_counter() - t0, 5), max_regret=regret, support=support))
            print(f"{game.name:28s} {name:20s} {'ok' if ok else 'FAIL':4s} {rows[-1]['seconds']:.4f}s")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary: mean time (solved instances) and success rate per solver, size and rank
    agg = defaultdict(list)
    for r in rows:
        agg[(r["solver"], r["n"], r["k"])].append(r)
    lines = ["| solver | n | k | solved | mean time (s) |", "|---|---|---|---|---|"]
    for (s, n, k), rs in sorted(agg.items()):
        solved = [r for r in rs if r["solved"]]
        mean_t = np.mean([r["seconds"] for r in solved]) if solved else float("nan")
        lines.append(f"| {s} | {n} | {k} | {len(solved)}/{len(rs)} | {mean_t:.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out/'results.csv'} and {out/'summary.md'}")


if __name__ == "__main__":
    main()
