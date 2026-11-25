"""
CE importance sampling for |Ham(G_n)|  — v2 with IS-weighted CE update.

Key fix: the unweighted CE update ("maximize completion rate") collapses
to a distribution concentrated on a tiny fraction of Ham paths, so IS
weights shrink and the count estimate collapses to zero.

The IS-WEIGHTED update weights each sample by 1/Q(π), giving more pull
from rare paths.  The fixed point is Q* ∝ I[Ham path] (uniform over Ham
paths), where completion rate → 1 and all IS weights equal N.
"""
import numpy as np
from math import isqrt, log
import time, sys
sys.path.insert(0, '/home/claude')
from ce_sampler import sample_batch

EXACT = {57: 68_092_497_615, 59: 332_409_010_079}

def build_adj(n):
    adj=[[] for _ in range(n+1)]
    for u in range(1,n+1):
        for v in range(u+1,n+1):
            s=u+v; r=isqrt(s)
            if r*r==s: adj[u].append(v); adj[v].append(u)
    return adj

def adj_to_arrays(adj, n):
    offsets=np.zeros(n+2,dtype=np.int32)
    for u in range(1,n+1): offsets[u+1]=offsets[u]+len(adj[u])
    flat=np.zeros(offsets[n+1],dtype=np.int32)
    for u in range(1,n+1): flat[offsets[u]:offsets[u+1]]=adj[u]
    return flat, offsets

def is_stats(log_qs, M, exact=None):
    if len(log_qs) == 0:
        return dict(n_done=0, rate=0., est=0., se=float('inf'),
                    cv=float('inf'), ess_frac=0., ratio=float('nan'))
    n_done  = len(log_qs)
    log_w   = np.asarray([-lq for lq in log_qs])
    max_lw  = log_w.max()
    log_est = max_lw + log(np.exp(log_w - max_lw).sum()) - log(M)
    est     = float(np.exp(log_est))
    w_pos   = np.exp(log_w)
    w_all   = np.zeros(M); w_all[:n_done] = w_pos
    se      = w_all.std(ddof=1) / np.sqrt(M)
    cv      = se / est if est > 0 else float('inf')
    ess     = (w_pos.sum()**2 / (w_pos**2).sum()) if n_done > 1 else 1.
    ess_frac= ess / M
    return dict(n_done=n_done, rate=n_done/M, est=est, se=se,
                cv=cv, ess_frac=ess_frac,
                ratio=est/exact if exact else float('nan'))

def ce_update_weighted(w, log_qs, path_data, n_done, n, adj, alpha=0.7, eps=1e-2):
    """
    IS-weighted CE update.  Each completed path contributes proportionally
    to its IS weight  w_i = 1/Q(π_i) = exp(-log_q_i).
    
    This moves Q towards uniform-over-Ham-paths (the correct counting IS
    target) rather than just maximising completion rate.
    
    To keep weights numerically stable we normalise:
      is_w_i = exp(log_w_i - max_log_w)   (relative weights, max=1)
    """
    if n_done == 0:
        return
    n_s   = n - 1
    log_w = np.array([-lq for lq in log_qs[:n_done]])  # log IS weights
    max_lw= log_w.max()
    is_w  = np.exp(log_w - max_lw)          # relative IS weights, max = 1

    cnt_at = np.zeros((n_s, n+1),      dtype=np.float64)
    cnt_tr = np.zeros((n_s, n+1, n+1), dtype=np.float64)

    for i in range(n_done):
        path = path_data[i*n:(i+1)*n]
        wi   = float(is_w[i])
        for k in range(n_s):
            u, v = int(path[k]), int(path[k+1])
            cnt_at[k, u]    += wi
            cnt_tr[k, u, v] += wi

    for u in range(1, n+1):
        deg = len(adj[u])
        if deg == 0: continue
        for k in range(n_s):
            denom = cnt_at[k, u] + eps * deg
            if denom < 1e-9: continue
            for v in adj[u]:
                new_w = (cnt_tr[k, u, v] + eps) / denom
                w[k, u, v] = alpha * w[k, u, v] + (1. - alpha) * new_w

def run_ce(n, n_iters=30, alpha=0.7, seed=0):
    adj    = build_adj(n)
    flat, offsets = adj_to_arrays(adj, n)
    n_s    = n - 1
    exact  = EXACT.get(n)

    # Warnsdorff init: prefer low-degree next vertex
    w = np.zeros((n_s, n+1, n+1), dtype=np.float64)
    for u in range(1, n+1):
        for v in adj[u]:
            wv = 1. / (len(adj[v]) + 0.1)
            for k in range(n_s):
                w[k, u, v] = wv

    print(f"\nn={n}  exact={exact:.4e}  |E|={sum(len(adj[u]) for u in range(1,n+1))//2}"
          f"  alpha={alpha}")
    hdr=(f"{'it':>3}  {'M':>8}  {'done':>6}  {'rate':>6}  "
         f"{'estimate':>12}  {'ratio':>7}  {'CV':>6}  {'ESS%':>5}  time")
    print(hdr); print('─'*len(hdr))

    rng_seed = seed
    results  = []

    for it in range(n_iters):
        prev_rate = results[-1]['rate'] if results else 0.
        if   prev_rate < 0.0002: M = 2_000_000
        elif prev_rate < 0.002:  M =   500_000
        elif prev_rate < 0.02:   M =   200_000
        elif prev_rate < 0.10:   M =    50_000
        else:                    M =    20_000

        rng_seed += 1
        t0 = time.perf_counter()
        lqs, pdata, n_done = sample_batch(flat, offsets, w, n, M, rng_seed)
        elapsed = time.perf_counter() - t0

        st = is_stats(lqs, M, exact)
        print(f"{it:3d}  {M:8d}  {st['n_done']:6d}  {st['rate']:5.3%}  "
              f"{st['est']:12.4e}  {st['ratio']:7.4f}  {st['cv']:6.3f}  "
              f"{st['ess_frac']:4.1%}  {elapsed:.1f}s")
        results.append(st)

        if n_done > 0:
            ce_update_weighted(w, lqs.tolist(), pdata, int(n_done), n, adj, 
                               alpha=alpha)

    return results

def warmup():
    n0=10; adj0=build_adj(n0); fl0,of0=adj_to_arrays(adj0,n0)
    w0=np.ones((n0-1,n0+1,n0+1),dtype=np.float64)
    sample_batch(fl0,of0,w0,n0,5,99)

if __name__ == '__main__':
    print("Warming up JIT...", end=' ', flush=True)
    warmup(); print("done")

    ns = [int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [57, 59]
    for n in ns:
        res = run_ce(n, n_iters=30, alpha=0.7, seed=n*100)
        good = [r for r in res if r['est'] > 0]
        if good:
            r = good[-1]
            print(f"\nFinal: {r['est']:.6e}  exact: {EXACT.get(n,'?'):.6e}  "
                  f"ratio: {r['ratio']:.4f}  CV: {r['cv']:.3f}")
