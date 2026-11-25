"""Compare exact counts H(n) with the prediction of Conjecture 5.3:

    log H(n) = log(n!/2) - ((n-1)/2) log n + C n + O(log n).

Reads data/b071983.txt (n, number of undirected Hamiltonian paths).

    python -m hamiltonian.bounds.residuals
"""
from __future__ import annotations

from math import lgamma, log
from pathlib import Path

import numpy as np

from .sinkhorn import constant_C

DATA = Path(__file__).resolve().parents[2] / "data" / "b071983.txt"


def main() -> None:
    C, _ = constant_C(1600)
    rows = [l.split() for l in DATA.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    ns, rs = [], []
    for n, h in rows:
        n, h = int(n), int(h)
        if h == 0:
            continue
        r = log(h) - (lgamma(n + 1) - log(2) - (n - 1) / 2 * log(n) + C * n)
        ns.append(n); rs.append(r)
        print(f"{n:3d}  {h:>24d}  residual {r:+.2f}")
    ns, rs = np.array(ns), np.array(rs)
    m = (ns >= 40) & (ns <= 74)
    A = np.vstack([np.log(ns[m]), np.ones(m.sum())]).T
    print("fit residual = a log n + b on 40..74:", np.linalg.lstsq(A, rs[m], rcond=None)[0])
    print(f"mean {rs[m].mean():+.3f}, sd {rs[m].std():.3f} on 40..74")


if __name__ == "__main__":
    main()
