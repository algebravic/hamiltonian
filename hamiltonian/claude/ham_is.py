#!/usr/bin/env python3
"""
ham_is.py  —  Importance-sampling approximation of |Ham(G_n)|

USAGE
-----
    python ham_is.py [n] [options]

    python ham_is.py 57                     # quick run, n=57
    python ham_is.py 63 --M 500000          # 500K samples
    python ham_is.py 80 --alpha 2.5         # tune Warnsdorff aggressiveness
    python ham_is.py 57 --truth 68092497615 # compare to exact value

OPTIONS
-------
    --M      M         Total IS samples        (default: 100 000)
    --alpha  α         Warnsdorff exponent      (default: 2.0)
    --seed   s         Random seed              (default: 42)
    --truth  N         Known undirected count   (optional, for z-score)
    --batch  b         Print running average every b samples  (default: 0 = off)
    --sweep             Sweep α ∈ {1,1.5,2,2.5,3,4,5} to find optimum

BACKGROUND
----------
IS distribution (Warnsdorff-weighted Markov chain on G_n):

    p(v | u, V) ∝ exp(log_w[u,v]) / rem_deg(v)^α

where rem_deg(v) is the number of unvisited neighbors of v at the current step.
α=2 minimises effective variance for both G_57 and G_59 empirically.

Unbiased estimator for *directed* Ham paths:

    N̂_dir = (1/M) Σ_i  I(path_i completes) / Q(path_i)

The DP counts *undirected* paths, so divide by 2:

    N̂_undir = N̂_dir / 2

Variance metric:
    eff_var = (1 + CV²) / cr
where CV = coefficient of variation of non-zero IS weights, cr = completion rate.
The number of samples needed for ε relative SEM ≈ eff_var / ε².

KNOWN EXACT VALUES (undirected, from profile DP)
-------------------------------------------------
    |Ham(G_57)| = 68 092 497 615
    |Ham(G_59)| = 332 409 010 079
    |Ham(G_61)| = 3 711 439 128 718
    |Ham(G_62)| = 19 133 360 717 991
    |Ham(G_63)| = 121 460 168 051 419
"""

import math, random, time, argparse
import numpy as np


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph(n: int) -> list[list[int]]:
    """Adjacency list for G_n: edge u-v iff u+v is a perfect square."""
    sq  = set(i*i for i in range(2, 2*n + 2))
    adj = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if u + v in sq:
                adj[u].append(v)
                adj[v].append(u)
    return adj


def graph_stats(n: int, adj: list) -> dict:
    degs = [len(adj[u]) for u in range(1, n + 1)]
    return dict(n_edges  = sum(degs) // 2,
                deg_min  = min(degs),
                deg_max  = max(degs),
                deg_mean = sum(degs) / n,
                isolated = sum(1 for d in degs if d == 0),
                leaves   = sum(1 for d in degs if d == 1))


# ── IS sampler ─────────────────────────────────────────────────────────────────

def sample_path(adj: list, n: int, alpha: float) -> tuple[list | None, float]:
    """
    Draw one directed Ham-path attempt under the Warnsdorff IS distribution.

    Returns (path, log_Q) if all n vertices are visited,
    or       (None, 0.0)  on dead-end.

    log_Q = log probability of this exact sequence under the IS distribution.
    IS weight = exp(-log_Q) = 1/Q(path).
    """
    visited = bytearray(n + 1)               # 0 = unvisited
    rem     = [len(adj[u]) for u in range(n + 1)]  # dynamic remaining degree

    # choose start uniformly
    s = random.randint(1, n)
    visited[s] = 1
    for nb in adj[s]:
        rem[nb] -= 1

    path  = [s]
    log_Q = -math.log(n)                     # log prob of start vertex

    for _ in range(n - 1):
        u    = path[-1]
        nbrs = [v for v in adj[u] if not visited[v]]
        if not nbrs:
            return None, 0.0                 # dead-end

        # Warnsdorff weights: w(v) ∝ rem_deg(v)^{-alpha}
        ws_raw = [rem[v] ** (-alpha) if rem[v] > 0 else 1.0 for v in nbrs]
        s2     = sum(ws_raw)
        ws     = [w / s2 for w in ws_raw]

        v      = random.choices(nbrs, weights=ws_raw)[0]
        idx    = nbrs.index(v)
        log_Q += math.log(ws[idx])

        visited[v] = 1
        for nb in adj[v]:
            rem[nb] -= 1
        path.append(v)

    return path, log_Q


# ── Estimator ──────────────────────────────────────────────────────────────────

def estimate(n:     int,
             M:     int   = 100_000,
             alpha: float = 2.0,
             seed:  int   = 42,
             truth: int   = 0,
             batch: int   = 0,
             verbose: bool = True) -> dict:
    """
    Run M IS samples and return a result dict with all statistics.

    truth : known undirected count (0 = unknown).
    batch : if > 0, print a running average every `batch` samples.
    """
    random.seed(seed)
    adj = build_graph(n)
    gs  = graph_stats(n, adj)

    if verbose:
        bar = "─" * 62
        print(f"\n{bar}")
        print(f"  G_{n}   α={alpha}   M={M:,}   seed={seed}")
        print(f"  edges={gs['n_edges']}  deg∈[{gs['deg_min']},{gs['deg_max']}]  "
              f"mean_deg={gs['deg_mean']:.2f}  isolated={gs['isolated']}")
        if truth:
            print(f"  truth (undirected) = {truth:,}")
        print(f"{bar}")

    iweights    = []
    completions = 0
    t0          = time.perf_counter()

    for i in range(M):
        path, lQ = sample_path(adj, n, alpha)
        if path is not None:
            iweights.append(math.exp(-lQ))
            completions += 1
        else:
            iweights.append(0.0)

        if batch and (i + 1) % batch == 0 and verbose:
            W_so_far  = np.array(iweights)
            N_dir_est = W_so_far.mean()
            N_und_est = N_dir_est / 2
            cr_so_far = completions / (i + 1)
            nz        = W_so_far[W_so_far > 0]
            cv        = float(nz.std(ddof=1) / nz.mean()) if len(nz) > 1 else math.inf
            ratio_str = f"  r={N_und_est/truth:.4f}" if truth else ""
            print(f"  [{i+1:>8,}]  cr={cr_so_far:.2%}  N̂/2={N_und_est:.4e}{ratio_str}  CV={cv:.3f}")

    elapsed = time.perf_counter() - t0
    W       = np.array(iweights)
    nz      = W[W > 0]

    N_dir   = float(W.mean())
    N_undir = N_dir / 2
    sem_dir = float(W.std(ddof=1) / math.sqrt(M))
    sem_und = sem_dir / 2
    cr      = completions / M
    cv      = float(nz.std(ddof=1) / nz.mean()) if len(nz) > 1 else math.inf
    eff_var = (1 + cv**2) / max(cr, 1e-12)
    ess     = int(M / (1 + cv**2)) if cv < math.inf else 0
    lo, hi  = N_undir - 1.96 * sem_und, N_undir + 1.96 * sem_und

    result = dict(
        n           = n,
        M           = M,
        alpha       = alpha,
        N_directed  = N_dir,
        N_undirected= N_undir,
        sem         = sem_und,
        ci_lo       = lo,
        ci_hi       = hi,
        completions = completions,
        cr          = cr,
        cv          = cv,
        eff_var     = eff_var,
        ess         = ess,
        elapsed     = elapsed,
        truth       = truth,
    )

    if verbose:
        _print_result(result)

    return result


def _print_result(r: dict) -> None:
    t = r['truth']
    print(f"\n  Completions  = {r['completions']:,}  ({r['cr']:.2%})")
    print(f"  N̂ (undir)   = {r['N_undirected']:.6e} ± {r['sem']:.3e}  (1σ SEM)")
    print(f"  95% CI       = [{r['ci_lo']:.4e}, {r['ci_hi']:.4e}]")
    if t:
        z  = (r['N_undirected'] - t) / r['sem']
        ok = "✓" if abs(z) < 1.96 else "✗"
        print(f"  truth        = {t:.6e}")
        print(f"  ratio        = {r['N_undirected']/t:.5f}   z={z:+.2f}σ  {ok} 95%CI")
    print(f"  CV={r['cv']:.3f}   ESS≈{r['ess']:,}   eff_var={r['eff_var']:.0f}")
    print(f"  → 5% accuracy ≈ {int(r['eff_var']/0.0025):,} samples")
    print(f"  → 1% accuracy ≈ {int(r['eff_var']/0.0001):,} samples")
    print(f"  time = {r['elapsed']:.2f}s  ({r['M']/r['elapsed']:.0f} samp/s)")
    print("─" * 62)


# ── Alpha sweep ────────────────────────────────────────────────────────────────

def alpha_sweep(n: int, M: int = 5_000, seed: int = 42,
                truth: int = 0,
                alphas: list[float] | None = None) -> None:
    """Run a quick α sweep to find the best Warnsdorff exponent for G_n."""
    if alphas is None:
        alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    adj = build_graph(n)
    gs  = graph_stats(n, adj)
    print(f"\nα sweep  G_{n}  M={M:,}")
    print(f"  edges={gs['n_edges']}  deg∈[{gs['deg_min']},{gs['deg_max']}]")
    print(f"  {'α':>5}  {'cr':>6}  {'N̂(undir)':>13}  {'ratio':>8}  {'CV':>7}  {'eff_var':>9}")

    best = None
    for alpha in alphas:
        random.seed(seed)
        iws = []
        completions = 0
        for _ in range(M):
            path, lQ = sample_path(adj, n, alpha)
            if path is not None:
                iws.append(math.exp(-lQ)); completions += 1
            else:
                iws.append(0.0)
        W      = np.array(iws); nz = W[W > 0]
        N_und  = W.mean() / 2
        cr     = completions / M
        cv     = float(nz.std(ddof=1) / nz.mean()) if len(nz) > 1 else math.inf
        eff    = (1 + cv**2) / max(cr, 1e-12)
        ratio  = f"{N_und/truth:.4f}" if truth and N_und > 0 else "     —"
        best_marker = ""
        if best is None or eff < best[1]:
            best = (alpha, eff)
            best_marker = " ← best"
        print(f"  {alpha:5.1f}  {cr:6.2%}  {N_und:13.4e}  {ratio:>8}  {cv:7.3f}  {eff:9.1f}{best_marker}")

    if best:
        print(f"\n  Recommended α ≈ {best[0]}  (eff_var={best[1]:.0f})")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Importance-sampling estimate of |Ham(G_n)|",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("n",           type=int,   nargs="?", default=57,
                   help="Graph size (default: 57)")
    p.add_argument("--M",         type=int,   default=100_000,
                   help="Number of IS samples (default: 100 000)")
    p.add_argument("--alpha",     type=float, default=2.0,
                   help="Warnsdorff exponent (default: 2.0)")
    p.add_argument("--seed",      type=int,   default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--truth",     type=int,   default=0,
                   help="Known undirected count for z-score (optional)")
    p.add_argument("--batch",     type=int,   default=0,
                   help="Print running average every BATCH samples (0=off)")
    p.add_argument("--sweep",     action="store_true",
                   help="Sweep α values to find optimum first")
    args = p.parse_args()

    # inject known exact values
    known = {57: 68_092_497_615, 59: 332_409_010_079,
             61: 3_711_439_128_718, 62: 19_133_360_717_991,
             63: 121_460_168_051_419}
    truth = args.truth or known.get(args.n, 0)

    if args.sweep:
        alpha_sweep(args.n, M=min(args.M, 5_000), seed=args.seed, truth=truth)

    estimate(n=args.n, M=args.M, alpha=args.alpha,
             seed=args.seed, truth=truth, batch=args.batch)


if __name__ == "__main__":
    main()
