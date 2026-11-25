"""Deterministic lower bounds for the number of cycle covers of G(n)
(Section 7 and Appendix A of doc/square_sum_bounds.tex).

per(A) counts permutations pi with i ~ pi(i) in G(n) (directed cycle covers,
2-cycles allowed).  For every doubly stochastic P supported on the edges,
Gurvits-Samorodnitsky give

    log per(A) >= F(P) = sum_{ij} [ -p_ij log p_ij + (1 - p_ij) log(1 - p_ij) ].

We take P = S, the symmetric Sinkhorn scaling D A D of A, and compare
(F(S) - (n/2) log n)/n with C - 1.  We also measure the row-sum error of the
continuum ansatz p_ij = a(i/n) a(j/n) / sqrt(n).

    python -m hamiltonian.bounds.bethe --n 1000 4000 16000
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import scipy.sparse as sp

from .sinkhorn import constant_C


def adjacency_sparse(n: int) -> sp.csr_matrix:
    rows, cols = [], []
    for i in range(1, n + 1):
        for k in range(math.isqrt(i) + 1, math.isqrt(i + n) + 1):
            j = k * k - i
            if j != i:
                rows.append(i - 1)
                cols.append(j - 1)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def sinkhorn_symmetric(A: sp.csr_matrix, iters: int = 3000) -> np.ndarray:
    x = np.ones(A.shape[0])
    for _ in range(iters):
        x = np.sqrt(x / (A @ x))
    return x


def bethe_value(P: sp.csr_matrix) -> float:
    p = P.data
    return float(np.sum(-p * np.log(p) + (1 - p) * np.log1p(-p)))


def continuum_potential(M: int = 3200) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(M) + 0.5) / M
    K = 1.0 / (2.0 * np.sqrt(xs[:, None] + xs[None, :]))
    a = np.ones(M)
    for _ in range(5000):
        a = np.sqrt(a / (K @ a / M))
    return xs, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1000, 4000, 16000])
    args = ap.parse_args()
    C, _ = constant_C(1600)
    xs, a = continuum_potential()
    print(f"C - 1 = {C - 1:+.5f}")
    for n in args.n:
        A = adjacency_sparse(n)
        x = sinkhorn_symmetric(A)
        S = sp.diags(x) @ A @ sp.diags(x)
        F = bethe_value(S.tocsr())
        av = np.interp((np.arange(1, n + 1)) / n, xs, a)
        ansatz_rows = av * (A @ av) / math.sqrt(n)
        print(f"n={n:7d}  (F(S)-(n/2)log n)/n = {(F - n / 2 * math.log(n)) / n:+.5f}   "
              f"ansatz max|rowsum-1| = {np.abs(ansatz_rows - 1).max():.4f}  "
              f"rms = {np.sqrt(np.mean((ansatz_rows - 1) ** 2)):.4f}")


if __name__ == "__main__":
    main()
