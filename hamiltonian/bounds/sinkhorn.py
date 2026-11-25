"""The constant C of Conjecture 5.3 in doc/square_sum_bounds.tex.

kappa(x, y) = 1 / (2 sqrt(x + y)) on [0,1]^2.  C = sup over couplings pi of
(Lebesgue, Lebesgue) of  int log kappa dpi - KL(pi | Lebesgue^2)  = -2 int log a,
where a > 0 solves  a(x) int_0^1 a(y) kappa(x, y) dy = 1  (Schroedinger system).

    python -m hamiltonian.bounds.sinkhorn
"""
from __future__ import annotations

import numpy as np


def constant_C(M: int, iters: int = 5000) -> tuple[float, float]:
    x = (np.arange(M) + 0.5) / M                     # midpoint rule
    K = 1.0 / (2.0 * np.sqrt(x[:, None] + x[None, :]))
    a = np.ones(M)
    for _ in range(iters):                            # damped symmetric Sinkhorn
        a = np.sqrt(a / (K @ a / M))
    resid = float(np.abs(a * (K @ a / M) - 1).max())
    return float(-2.0 * np.mean(np.log(a))), resid


def main() -> None:
    for M in (400, 1600, 3200):
        C, r = constant_C(M)
        print(f"M={M:5d}  C={C:.7f}  residual={r:.1e}")


if __name__ == "__main__":
    main()
