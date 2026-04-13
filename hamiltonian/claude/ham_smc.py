#!/usr/bin/env python3
"""
ham_smc.py — Sequential Monte Carlo (particle filter) estimate of |Ham(G_n)|

VERTEX ORDERING
---------------
Uses best-BFS ordering: tries every BFS root, picks the one minimising the
maximum frontier width.  For G_57 this gives max-frontier ≈ 19 (natural order
is 40; exact pathwidth is 14).  Results are correct but high-variance for
n ≳ 35; use --reps R for reliable estimates.

For large n the C profile-DP is required as a guide; see the DP-guided section
in the accompanying docstring.

STATE PER PARTICLE
------------------
  paired[u]   = other frontier endpoint of u's chain, or None if that end
                has already departed as a global path-endpoint
  free        = frontier vertices with degree 0
  deg         = degree of every tracked frontier vertex (0|1|2)
  n_chains    = chains created minus merges
  closed      = chains with zero remaining frontier endpoints (permanently frozen)
  global_eps  = vertices that departed as path endpoints (must equal 2)
  weight      = product of local branching factors

VALIDITY CONSTRAINTS ON S
--------------------------
  (A) |S| ≤ 2
  (B) no-cycle: S must not contain both endpoints of the same chain
  Forced type 1: deg-0 vertices with last-future-neighbour == v  (isolated otherwise)
  Forced type 2: deg-1 vertices with last-future-neighbour == v, when enough
                 are present that letting them all depart would push global_eps > 2

PRUNING
-------
  When a chain becomes permanently frozen (closed += 1) and n_chains > 1,
  the particle is killed immediately (can never merge all chains into one).

ESTIMATE
--------
  N̂ = (1/M) Σ_{valid-final p} w_p
  
  Valid final: n_chains==1, closed==1, global_eps==2, free=={}, paired=={}.

USAGE
-----
  python ham_smc.py 15                           # n=15, truth=1
  python ham_smc.py 57 --M 5000 --reps 10       # multi-run, get variance
  python ham_smc.py 57 --M 1000 --no-resample   # pure IS (no resampling)
  python ham_smc.py 25 --M 2000 --reps 5

KNOWN EXACT VALUES (undirected)
  n=15  → 1          n=57  → 68 092 497 615
  n=25  → 10         n=59  → 332 409 010 079
  n=30  → 20         n=61  → 3 711 439 128 718
                     n=62  → 19 133 360 717 991
                     n=63  → 121 460 168 051 419

DP-GUIDED IS (CONCEPTUAL)
--------------------------
The variance of this SMC comes from random branching-factor choices at each
step.  The ideal proposal: at each step, sample S proportionally to the exact
DP count of Ham paths consistent with the resulting frontier state.  Under
this "DP-guided" proposal every IS weight equals 1 and all M particles reach
valid terminal states.

With the existing C DP (ham_dp_c.py), this can be implemented by:
  1. Run the C DP normally.  Expose a query C(state_key, step) returning the
     number of completions from that frontier state.
  2. In the SMC: for each valid S, apply it to the particle, compute the
     resulting state key (using the same 4-bits-per-slot encoding), and call
     C(state_key, step+1).
  3. Sample S proportional to those counts.  Update IS weight by
     w *= (sum of C values over all valid S) / C(chosen result state).
     (This ratio equals 1 if counts are exact.)

Under this scheme the particle population IS the profile DP, just sampled
instead of enumerated.  The particle count M replaces the exponential state
space, and estimates converge as O(1/√M) regardless of n.
"""

import math, random, time, argparse
from collections import deque
from itertools import combinations
import numpy as np


# ── Graph + ordering ───────────────────────────────────────────────────────────

def build_graph(n):
    sq = set(i*i for i in range(2, 2*n+2))
    adj = [[] for _ in range(n+1)]
    for u in range(1, n+1):
        for v in range(u+1, n+1):
            if u+v in sq:
                adj[u].append(v); adj[v].append(u)
    return adj


def best_bfs_order(n, adj):
    """
    Try every BFS root; return (order, max_frontier_width).
    Relabels vertices so the natural 1..n order matches this BFS ordering.
    Returns: (adj_new, max_fw)  where adj_new is the relabeled adjacency list.
    """
    def mfw_of(order):
        pos = [0]*(n+1)
        for i,v in enumerate(order): pos[v] = i+1  # 1-indexed positions
        delta = [0]*(n+2)
        for u in range(1, n+1):
            fut = [pos[nb] for nb in adj[u] if pos[nb] > pos[u]]
            if fut:
                delta[pos[u]] += 1
                delta[max(fut)+1] -= 1
        fw=0; mw=0
        for i in range(1, n+1): fw += delta[i]; mw = max(mw, fw)
        return mw

    best_mfw = n+1; best_order = list(range(1, n+1))
    for root in range(1, n+1):
        vis = bytearray(n+1); order = []; q = deque([root]); vis[root] = 1
        while q:
            v = q.popleft(); order.append(v)
            for nb in adj[v]:
                if not vis[nb]: vis[nb]=1; q.append(nb)
        for v in range(1, n+1):
            if not vis[v]: order.append(v)
        mw = mfw_of(order)
        if mw < best_mfw:
            best_mfw = mw; best_order = list(order)

    # Relabel: best_order[0] → 1, best_order[1] → 2, ...
    relabel = {v: i+1 for i, v in enumerate(best_order)}
    adj_new = [[] for _ in range(n+1)]
    for u in range(1, n+1):
        nu = relabel[u]
        for nb in adj[u]:
            adj_new[nu].append(relabel[nb])

    return adj_new, best_mfw, best_order[0]   # (new adj, max_fw, BFS root)


def precompute(n, adj):
    max_future = [0]*(n+1)
    for u in range(1, n+1):
        fut = [v for v in adj[u] if v > u]
        max_future[u] = max(fut) if fut else 0
    leaves_at = [[] for _ in range(n+1)]
    for u in range(1, n+1):
        lv = max_future[u] if max_future[u] > 0 else u
        leaves_at[lv].append(u)
    return max_future, leaves_at


# ── Particle ───────────────────────────────────────────────────────────────────

class Particle:
    """
    Frontier state.  closed = number of permanently frozen chains.
    Particle is killed when closed >= 1 and n_chains > 1.
    """
    __slots__ = ['paired','free','deg','n_chains','closed','global_eps','weight']

    def __init__(self):
        self.paired    = {}
        self.free      = set()
        self.deg       = {}
        self.n_chains  = 0
        self.closed    = 0
        self.global_eps= 0
        self.weight    = 1.0

    def clone(self):
        q = Particle.__new__(Particle)
        q.paired     = dict(self.paired)
        q.free       = set(self.free)
        q.deg        = dict(self.deg)
        q.n_chains   = self.n_chains
        q.closed     = self.closed
        q.global_eps = self.global_eps
        q.weight     = self.weight
        return q

    def is_valid_final(self):
        # Note: closed is NOT checked here. For a long chain, both global
        # endpoints depart through non-None partners, so closed stays 0.
        # closed is only used for early-exit pruning during the DP sweep.
        return (not self.free and not self.paired
                and self.global_eps == 2
                and self.n_chains == 1)

    def _no_cycle(self, S):
        eps = [u for u in S if self.deg.get(u,0) == 1]
        for i in range(len(eps)):
            for j in range(i+1, len(eps)):
                if self.paired.get(eps[i]) == eps[j]:
                    return False
        return True

    def _valid_transitions(self, v, adj_v, max_future):
        past_active = [u for u in adj_v
                       if u < v and u in self.deg and self.deg[u] < 2]
        forced_A = [u for u in past_active
                    if self.deg[u] == 0 and max_future[u] == v]
        dep_ep   = [u for u in past_active
                    if self.deg[u] == 1 and max_future[u] == v]
        allowed      = max(0, 2 - self.global_eps)
        must_connect = max(0, len(dep_ep) - allowed)
        if len(forced_A) + must_connect > 2:
            return []
        optional = [u for u in past_active if u not in forced_A and u not in dep_ep]
        slots_A  = 2 - len(forced_A)
        valid = []
        for ep_k in range(must_connect, min(len(dep_ep), slots_A)+1):
            for ep_sub in combinations(dep_ep, ep_k):
                slots_opt = slots_A - ep_k
                for opt_j in range(slots_opt+1):
                    for opt_sub in combinations(optional, opt_j):
                        S = forced_A + list(ep_sub) + list(opt_sub)
                        if self._no_cycle(S):
                            valid.append(S)
        return valid

    def _connect(self, v, S):
        dv = len(S); self.deg[v] = dv
        if dv == 0:
            self.free.add(v); return
        if dv == 1:
            u = S[0]
            if self.deg[u] == 0:
                self.free.discard(u); self.deg[u]=1
                self.paired[u]=v; self.paired[v]=u
                self.n_chains += 1
            else:
                p = self.paired.pop(u); self.deg[u]=2
                self.paired[v]=p
                if p is not None: self.paired[p]=v
        else:
            self.deg[v]=2; u,w = S[0],S[1]; du,dw = self.deg[u],self.deg[w]
            if du==0 and dw==0:
                self.free.discard(u); self.free.discard(w)
                self.deg[u]=1; self.deg[w]=1
                self.paired[u]=w; self.paired[w]=u; self.n_chains+=1
            elif du==0:
                self.free.discard(u); pw=self.paired.pop(w)
                self.deg[u]=1; self.deg[w]=2; self.paired[u]=pw
                if pw is not None: self.paired[pw]=u
            elif dw==0:
                self.free.discard(w); pu=self.paired.pop(u)
                self.deg[w]=1; self.deg[u]=2; self.paired[w]=pu
                if pu is not None: self.paired[pu]=w
            else:
                pu=self.paired.pop(u); pw=self.paired.pop(w)
                self.deg[u]=2; self.deg[w]=2; self.n_chains-=1
                if pu is not None and pw is not None:
                    self.paired[pu]=pw; self.paired[pw]=pu
                elif pu is not None: self.paired[pu]=None
                elif pw is not None: self.paired[pw]=None

    def _depart(self, u):
        """Returns False if particle must die."""
        d = self.deg.pop(u, None)
        if d is None: return True
        if d == 0: self.free.discard(u); return False   # isolated
        if d == 1:
            self.global_eps += 1
            partner = self.paired.pop(u, None)
            if partner is not None:
                self.paired[partner] = None
            else:
                # u was the LAST frontier endpoint of its chain → chain is now frozen
                self.closed += 1
                if self.closed >= 1 and self.n_chains > 1:
                    return False   # can never merge all chains → kill
        return True

    def step(self, v, adj_v, max_future, leaves_at_v):
        valid = self._valid_transitions(v, adj_v, max_future)
        if not valid:
            self.weight = 0.0; return False
        S = random.choice(valid)
        self.weight *= len(valid)
        self._connect(v, S)
        for u in leaves_at_v:
            if not self._depart(u):
                self.weight = 0.0; return False
        return True


# ── SMC ────────────────────────────────────────────────────────────────────────

def _systematic_resample(particles, M):
    weights = [p.weight for p in particles]
    W = sum(weights)
    if W == 0: return [Particle() for _ in range(M)], 0.0
    step = W/M; u0 = random.uniform(0.0, step)
    cumw=0.0; j=0; new=[]
    for i in range(M):
        target = u0 + i*step
        while j < M-1 and cumw+weights[j] < target:
            cumw+=weights[j]; j+=1
        q=particles[j].clone(); q.weight=step; new.append(q)
    return new, W


def smc_estimate(n, M=1000, thresh=0.5, seed=42, truth=0,
                 verbose=True, resample=True):
    random.seed(seed)
    adj_raw  = build_graph(n)
    adj, mfw, bfs_root = best_bfs_order(n, adj_raw)
    max_future, leaves_at = precompute(n, adj)

    if verbose:
        print(f"\nSMC  G_{n}  M={M}  thresh={thresh}  seed={seed}"
              + (f"  truth={truth}" if truth else ""))
        print(f"  BFS root={bfs_root}  max_frontier={mfw}")

    particles = [Particle() for _ in range(M)]
    resample_count = 0; t0 = time.perf_counter()

    for v in range(1, n+1):
        alive = 0
        for p in particles:
            if p.step(v, adj[v], max_future, leaves_at[v]):
                alive += 1
        weights = [p.weight for p in particles]
        W = sum(weights)
        if W == 0:
            if verbose: print(f"  step {v:3d}: ALL DEAD")
            return dict(n=n,M=M,N_undirected=0.0,sem=0.0,valid=0,
                        resample_count=0,elapsed=time.perf_counter()-t0,truth=truth)
        W2  = sum(w*w for w in weights)
        ess = W*W/W2 if W2>0 else 0.0
        do_res = resample and ess < thresh*M
        if verbose and (v<=5 or v%10==0 or v>=n-2 or do_res):
            print(f"  step {v:3d}: alive={alive:>5}/{M}  ESS={ess:>8.1f}"
                  f"  W̄={W/M:.4e}" + ("  ←" if do_res else ""))
        if do_res:
            particles,_ = _systematic_resample(particles, M); resample_count+=1

    valid_w = [p.weight for p in particles if p.is_valid_final()]
    all_w   = [p.weight if p.is_valid_final() else 0.0 for p in particles]
    N_hat   = sum(valid_w)/M
    sem     = float(np.std(all_w, ddof=1)/math.sqrt(M))
    result  = dict(n=n, M=M, thresh=thresh, seed=seed, truth=truth,
                   N_undirected=N_hat, sem=sem, valid=len(valid_w),
                   resample_count=resample_count,
                   elapsed=time.perf_counter()-t0)
    if verbose: _print_result(result)
    return result


def _print_result(r):
    t=r['truth']; N=r['N_undirected']; se=r['sem']
    bar="─"*62
    print(f"\n{bar}")
    print(f"  N̂ (undirected) = {N:.6e} ± {se:.3e}  (use --reps for CI)")
    if t and se > 0:
        z=(N-t)/se; ok="✓" if abs(z)<1.96 else "✗"
        print(f"  truth={t:.6e}  ratio={N/t:.5f}  z={z:+.2f}σ  {ok}")
    print(f"  valid={r['valid']}/{r['M']}  resamples={r['resample_count']}")
    print(f"  time={r['elapsed']:.2f}s")
    print(bar)


KNOWN = {15:1, 25:10, 30:20,
         57:68_092_497_615, 59:332_409_010_079,
         61:3_711_439_128_718, 62:19_133_360_717_991,
         63:121_460_168_051_419}


def multi_run(n, M=1000, thresh=0.5, R=5, seed=42, truth=0, resample=True):
    truth=truth or KNOWN.get(n,0)
    bar="═"*62
    print(f"\n{bar}"); print(f"  SMC G_{n}  M={M}  thresh={thresh}  R={R}"); print(bar)
    estimates=[]
    for r in range(R):
        s=seed+r*137
        rs=smc_estimate(n,M=M,thresh=thresh,seed=s,truth=0,verbose=False,resample=resample)
        Ni=rs['N_undirected']; estimates.append(Ni)
        ratio=f"  r={Ni/truth:.5f}" if truth and Ni>0 else ""
        print(f"  run {r+1:>2}: N̂={Ni:.5e}{ratio}  t={rs['elapsed']:.1f}s"
              f"  valid={rs['valid']}/{M}")
    E=np.array(estimates); mu=E.mean(); std=E.std(ddof=1)
    cv=std/mu if mu>0 else float('inf'); sem=std/math.sqrt(R)
    print(f"\n  mean={mu:.6e}  std={std:.3e}  CV={cv:.4f}  SEM={sem:.3e}")
    if truth:
        z=(mu-truth)/sem if sem>0 else float('inf')
        ok="✓" if abs(z)<1.96 else "✗"
        print(f"  truth={truth:.6e}  ratio={mu/truth:.5f}  z={z:+.2f}σ  {ok}")
    print(bar)


def main():
    p=argparse.ArgumentParser(description="SMC estimate of |Ham(G_n)|",
                               formatter_class=argparse.RawDescriptionHelpFormatter,
                               epilog=__doc__)
    p.add_argument("n",            type=int,   nargs="?", default=57)
    p.add_argument("--M",          type=int,   default=1000)
    p.add_argument("--thresh",     type=float, default=0.5)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--truth",      type=int,   default=0)
    p.add_argument("--reps",       type=int,   default=1)
    p.add_argument("--no-resample",action="store_true")
    args=p.parse_args()
    truth=args.truth or KNOWN.get(args.n,0)
    resample=not args.no_resample
    if args.reps>1:
        multi_run(args.n,M=args.M,thresh=args.thresh,R=args.reps,
                  seed=args.seed,truth=truth,resample=resample)
    else:
        smc_estimate(args.n,M=args.M,thresh=args.thresh,
                     seed=args.seed,truth=truth,resample=resample)

if __name__=="__main__": main()
