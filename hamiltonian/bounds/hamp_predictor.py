"""Finite-n predictor for H(n) from the Sinkhorn scaling of G(n)
(Section 7.6 of doc/square_sum_bounds.tex).

Exact identity (Barvinok's Hamiltonian permanent, path version): with S = DAD
doubly stochastic and X a stationary Markov chain with transition matrix S,

    #directed Ham paths = n * prod(d_j)^{-2} * E[ d_{X_1} d_{X_n} ; X_1..X_n distinct ].

Zeroth-order ("major arc") approximation: replace P(distinct) by the i.i.d.
value n!/n^n and decouple the endpoints:

    Hhat(n) = (n/2) * (n!/n^n) * prod(d_j)^{-2} * (mean d)^2.

    python -m hamiltonian.bounds.hamp_predictor
"""
from __future__ import annotations

from math import lgamma, log
from pathlib import Path

import numpy as np

from .squares import adjacency

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "b071983.txt"
FIG = ROOT / "doc" / "Papers" / "figs" / "predictor.pdf"
C = -0.632538


def dense_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n))
    for i, js in adjacency(n).items():
        for j in js:
            A[i - 1, j - 1] = 1.0
    return A


def sinkhorn(A: np.ndarray, iters: int = 40000) -> np.ndarray:
    x = np.ones(len(A))
    for _ in range(iters):
        x = np.sqrt(x / (A @ x))
    return x


def log_hhat(n: int, d: np.ndarray) -> float:
    return log(n / 2) + lgamma(n + 1) - n * log(n) - 2 * np.sum(np.log(d)) + 2 * log(np.mean(d))


def main() -> None:
    H = {int(a): int(b) for a, b in (l.split() for l in DATA.read_text().splitlines() if l.strip())}
    rows = []
    for n in sorted(H):
        if H[n] == 0 or n < 25:
            continue
        A = dense_adjacency(n)
        d = sinkhorn(A)
        S = d[:, None] * A * d[None, :]
        err = float(np.abs(S.sum(1) - 1).max())
        r_old = log(H[n]) - (lgamma(n + 1) - log(2) - (n - 1) / 2 * log(n) + C * n)
        r_new = log(H[n]) - log_hhat(n, d)
        lam = np.sort(np.linalg.eigvalsh(S))[:-1]
        ls = np.arange(1, n)
        pair = float(np.sum((1 - ls / n) * np.array([np.sum(lam ** l) for l in ls])))
        logdet = float(np.sum(np.log(1 - lam)))
        rows.append((n, r_old, r_new, -pair, logdet, err))
        print(f"{n:3d} r_old={r_old:+.2f} r_scaling={r_new:+.2f}  "
              f"[pair-collision corr={-pair:+.2f}, log det'(I-S)={logdet:+.2f}]  rowerr={err:.0e}")
    good = np.array([r for r in rows if r[5] < 1e-10])
    print(f"converged ({int(good[0,0])}..{int(good[-1,0])}): "
          f"r_old sd={good[:,1].std():.3f}; r_scaling mean={good[:,2].mean():.3f} sd={good[:,2].std():.3f} "
          f"range [{good[:,2].min():.2f},{good[:,2].max():.2f}]")
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42      # embed TrueType, no Type 3 fonts
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
        allr = np.array([r for r in rows if r[5] < 1e-10])
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        ax.plot(allr[:, 0], allr[:, 1], "o-", ms=3, label="constant-$C$ formula")
        ax.plot(allr[:, 0], allr[:, 2], "s-", ms=3, label="scaling predictor $\\hat H(n)$")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("$n$"); ax.set_ylabel("$\\log H(n)-$prediction")
        ax.legend(fontsize=8)
        fig.tight_layout()
        FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG)
        print("figure:", FIG)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
