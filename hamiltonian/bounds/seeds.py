"""Search for seeds: Hamiltonian 1->3 paths of G(n), n odd, with many
separated parity-preserving moves.

    python -m hamiltonian.bounds.seeds --lo 25 --hi 43 --exhaustive
    python -m hamiltonian.bounds.seeds --lo 43 --hi 201 --samples 5 --window 24
"""
from __future__ import annotations

import argparse
import random

from .moves import max_separated, shortest_windows
from .squares import adjacency, all_paths, random_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=25)
    ap.add_argument("--hi", type=int, default=43)
    ap.add_argument("--exhaustive", action="store_true",
                    help="enumerate all 1->3 paths (feasible for n <= ~41)")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--window", type=int, default=0, help="max window length (0 = n)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    random.seed(a.seed)
    lo = a.lo | 1
    for n in range(lo, a.hi, 2):
        adj = adjacency(n)
        W = a.window or n
        paths = (all_paths(n, 1, 3) if a.exhaustive else
                 [p for p in (random_path(n, 1, 3, adj) for _ in range(a.samples)) if p])
        best = max((len(max_separated(shortest_windows(p, adj, W))) for p in paths),
                   default=0)
        print(f"n={n} paths={len(paths)} best k={best} k/(n+1/2)={best/(n+.5):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
