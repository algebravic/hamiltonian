"""Checks for Section 4 of doc/square_sum_bounds.tex.

e(G(n)) ~ (2/3)(sqrt 2 - 1) n^{3/2}, and the induced subgraph on the
multiples of 4 has twice the number of edges predicted by the kernel
kappa(x, y) = 1/(2 sqrt(x+y)) (Proposition 4.2).

    python -m hamiltonian.bounds.edges
"""
from __future__ import annotations

from math import isqrt, sqrt


def edges(n: int) -> int:
    """Number of edges of G(n): sum over squares s of #{i < j <= n : i + j = s}."""
    tot = 0
    for k in range(2, isqrt(2 * n - 1) + 1):
        s = k * k
        lo, hi = max(1, s - n), (s - 1) // 2          # i in [lo, hi], j = s - i > i
        tot += max(0, hi - lo + 1)
    return tot


def main() -> None:
    c0 = 2 / 3 * (sqrt(2) - 1)
    for n in (10**4, 10**6, 10**8):
        e, e4 = edges(n), edges(n // 4)      # multiples of 4 <-> G(n/4) via a -> 4a
        print(f"n={n:.0e}: e/n^1.5={e / n**1.5:.6f} (c0={c0:.6f}); "
              f"e(mult. of 4)/(e/16) = {e4 / (e / 16):.4f}")


if __name__ == "__main__":
    main()
