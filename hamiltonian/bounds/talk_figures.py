"""Figures for doc/talk/square_sum_talk.tex (written to doc/talk/figs/).

    python -m hamiltonian.bounds.talk_figures
"""
from __future__ import annotations

from math import lgamma, log
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bregman import B, log_upper_bound
from .sinkhorn import constant_C
from .squares import issq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "doc" / "talk" / "figs"
DATA = ROOT / "data" / "b071983.txt"


def adjacency_matrix(n: int) -> np.ndarray:
    i = np.arange(1, n + 1)
    s = i[:, None] + i[None, :]
    r = np.floor(np.sqrt(s)).astype(int)
    return ((r * r == s) & (i[:, None] != i[None, :])).astype(float)


def fig_adjacency(n: int = 400) -> None:
    A = adjacency_matrix(n)
    order = sorted(range(n), key=lambda k: ((k + 1) % 4, k))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(A, cmap="Greys", interpolation="nearest")
    ax[0].set_title(f"$G({n})$, natural order")
    ax[1].imshow(A[np.ix_(order, order)], cmap="Greys", interpolation="nearest")
    for b in (n // 4, n // 2, 3 * n // 4):
        ax[1].axhline(b, color="C3", lw=0.6)
        ax[1].axvline(b, color="C3", lw=0.6)
    ax[1].set_title("sorted by residue mod 4 (0,1,2,3)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "adjacency.pdf")
    plt.close(fig)


def fig_blocks(n: int = 4000) -> None:
    """Edge density between residue classes mod 4, relative to kappa."""
    A = adjacency_matrix(n)
    cls = [np.arange(n)[(np.arange(1, n + 1) % 4) == r] for r in range(4)]
    tot = A.sum() / n**2
    M = np.array([[A[np.ix_(cls[r], cls[s])].mean() / tot for s in range(4)]
                  for r in range(4)])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
    for r in range(4):
        for s in range(4):
            ax.text(s, r, f"{M[r, s]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xlabel("$j$ mod 4"); ax.set_ylabel("$i$ mod 4")
    ax.set_title("relative edge density", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "blocks_mod4.pdf")
    plt.close(fig)


def fig_kernel(M: int = 400) -> None:
    x = (np.arange(M) + 0.5) / M
    K = 1.0 / (2.0 * np.sqrt(x[:, None] + x[None, :]))
    a = np.ones(M)
    for _ in range(3000):
        a = np.sqrt(a / (K @ a / M))
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    im = ax[0].imshow(K, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmax=2)
    ax[0].set_title("$\\kappa(x,y)=1/(2\\sqrt{x+y})$")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].plot(x, a, label="$a(x)$")
    ax[1].plot(x, np.sqrt(1 + x) - np.sqrt(x), "--", label="degree profile $f(x)$")
    ax[1].legend(fontsize=8); ax[1].set_xlabel("$x$")
    ax[1].set_title("Schrödinger potential $a(x)$")
    mu = a[:, None] * a[None, :] * K
    im2 = ax[2].imshow(mu, origin="lower", extent=[0, 1, 0, 1], cmap="magma", vmax=2.5)
    ax[2].set_title("optimal coupling $a(x)a(y)\\kappa(x,y)$")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "kernel.pdf")
    plt.close(fig)


def load_counts():
    rows = [l.split() for l in DATA.read_text().splitlines() if l.strip()]
    return [(int(n), int(h)) for n, h in rows if int(h) > 0]


def fig_counts() -> None:
    C, _ = constant_C(1600)
    data = load_counts()
    ns = np.array([n for n, _ in data])
    norm = np.array([(log(h) - (lgamma(n + 1) - n / 2 * log(n))) / n for n, h in data])
    bn = np.arange(15, 3001, 5)
    bnorm = [(log_upper_bound(int(n)) - (lgamma(n + 1) - n / 2 * log(n))) / n for n in bn]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(ns[ns < 75], [log(h) for n, h in data if n < 75], "o", ms=3, label="exact")
    m = np.arange(15, 76)
    pred = [lgamma(k + 1) - log(2) - (k - 1) / 2 * log(k) + C * k for k in m]
    ax[0].plot(m, pred, "-", label="conjecture")
    ax[0].plot(75, log(dict(data)[75]), "x", color="C3", ms=7, label="$n=75$")
    ax[0].set_xlabel("$n$"); ax[0].set_ylabel("$\\log H(n)$"); ax[0].legend(fontsize=8)
    ax[1].plot(bn, bnorm, "-", color="C2", label="Bregman upper bound (exact)")
    ax[1].plot(ns[ns < 75], norm[ns < 75], "o", ms=3, label="exact counts")
    ax[1].axhline(B, color="C2", ls="--", lw=0.8, label=f"$B={B:.4f}$")
    ax[1].axhline(C, color="C1", ls="--", lw=0.8, label=f"$C={C:.4f}$")
    ax[1].set_xscale("log"); ax[1].set_xlabel("$n$")
    ax[1].set_ylabel("$(\\log H - \\log n! + \\frac{n}{2}\\log n)/n$")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "counts.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_adjacency(); fig_blocks(); fig_kernel(); fig_counts()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
