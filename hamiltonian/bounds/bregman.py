"""Rigorous upper bound for H(n) via the Bregman-Minc inequality.

Add a vertex 0 adjacent to everything; each directed Hamiltonian path
v_1..v_n of G(n) gives the cycle cover 0 -> v_1 -> ... -> v_n -> 0 of the
augmented graph, so 2 H(n) <= per(A') <= prod_i ((r_i)!)^{1/r_i}
(Bregman 1973), r_i = deg(i)+1 for i >= 1 and r_0 = n.  Asymptotically
deg(i) = sqrt(n) f(i/n) + O(1), f(x) = sqrt(1+x) - sqrt(x), so

    log H(n) <= (n/2) log n + (B - 1) n + o(n),
    B = int_0^1 log f = sqrt(2)/2 - (3/2) log(1 + sqrt 2) = -0.6149536...

(substitute x = sinh^2 t, so f = e^{-t}).  The conjectured constant C of
sinkhorn.py satisfies C <= B (Gibbs variational principle).

    python -m hamiltonian.bounds.bregman
"""
from __future__ import annotations

from math import lgamma, log, sqrt

from math import isqrt

B = sqrt(2) / 2 - 1.5 * log(1 + sqrt(2))


def degree(i: int, n: int) -> int:
    """deg of i in G(n): squares k^2 in [i+1, i+n], excluding k^2 = 2i."""
    d = isqrt(i + n) - isqrt(i)
    r = isqrt(2 * i)
    return d - (1 if r * r == 2 * i else 0)


def log_upper_bound(n: int) -> float:
    """Exact (finite n) Bregman bound for log H(n)."""
    degs = (degree(i, n) for i in range(1, n + 1))
    return sum(lgamma(d + 2) / (d + 1) for d in degs) + lgamma(n + 1) / n - log(2)


def main() -> None:
    print(f"B = {B:.7f}")
    for n in (25, 50, 75, 100, 1000, 10000):
        ub = log_upper_bound(n)
        norm = (ub - (lgamma(n + 1) - n / 2 * log(n))) / n
        print(f"n={n:6d}  log-bound={ub:10.2f}  (bound - log n! + (n/2)log n)/n = {norm:+.4f}")


if __name__ == "__main__":
    main()
