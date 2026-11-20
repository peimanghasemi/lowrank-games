"""Export the benchmark instances to instances/ as .npz files plus a manifest.

The instances are fully determined by (family, m, n, k, seed), so this is only
needed for sharing them with other codes (e.g. Julia or C++).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from lowrank_games import benchmark_set

p = argparse.ArgumentParser()
p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 50])
p.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
p.add_argument("--families", nargs="+", default=["uniform", "integer"])
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--out", default="instances")
a = p.parse_args()

out = Path(a.out)
out.mkdir(exist_ok=True)
with open(out / "manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "family", "m", "n", "k", "seed"])
    for g in benchmark_set(a.sizes, a.ranks, a.families, range(a.seeds)):
        np.savez_compressed(out / f"{g.name}.npz", A=g.A, B=g.B)
        w.writerow([g.name, g.meta["family"], g.meta["m"], g.meta["n"], g.meta["k"], g.meta["seed"]])
print(f"instances written to {out}/")
