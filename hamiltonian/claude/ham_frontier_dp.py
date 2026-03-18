"""
ham_frontier_dp.py
------------------
Direct profile / frontier DP for counting Hamiltonian paths in G_n.

State encoding
--------------
Each state is a tuple  (*connectivity, n_closed)  where:

  connectivity[i] for the i-th frontier vertex:
      0   : unvisited (fresh, no path edge yet)
      L>0 : endpoint of partial path-segment L
            L appears TWICE  → both endpoints still in frontier
            L appears ONCE   → one endpoint is external (already left frontier)
     -1   : interior node (degree 2 in selected edges; cannot take more)

  n_closed ∈ {0, 1} : number of path-segments where BOTH endpoints have
                        already left the frontier.  We immediately discard
                        states where n_closed would exceed 1.

Bug fixed vs naive DP
---------------------
A naive DP canonicalises only the connectivity tuple and loses track of
"closed chains" (components whose both endpoints became external at an
intermediate step).  This causes spurious multi-chain configurations to
accumulate silently and get counted as valid Hamiltonian paths.

The fix: detect the two events that turn a chain into a complete (closed)
segment and increment n_closed:

  1. *Elimination*: vertex u exits the frontier and u is the last in-frontier
     endpoint of its chain (su > 0 and su ∉ remaining connectivity tuple).
  2. *Half-external merge*: both sv and sw appear exactly ONCE in the state
     (each chain has one external endpoint); connecting them makes both
     interior so their two external endpoints form a closed chain.

In both cases we discard the state if n_closed > 1 (second closed chain)
or if there are still open/fresh vertices in the frontier that could not
be incorporated (n_closed = 1 at intermediate step with no path to grow).

Public API
----------
count_hamiltonian_paths(n, order=None, checkpoint_dir=None, verbose=False)
    -> int
"""

from math import isqrt
import pickle
import os
from collections import defaultdict

from ham_ordering import build_graph, best_order, frontier_stats


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _canonicalize(state: tuple) -> tuple:
    """Relabel positive integers in connectivity (all but last) in appearance order."""
    conn = state[:-1]
    nc = state[-1]
    mapping = {}
    nxt = 1
    result = []
    for x in conn:
        if x <= 0:
            result.append(x)
        else:
            if x not in mapping:
                mapping[x] = nxt
                nxt += 1
            result.append(mapping[x])
    return tuple(result) + (nc,)


def _max_label(state: tuple) -> int:
    return max((x for x in state[:-1] if x > 0), default=0)


def _count_label(state: tuple, L: int) -> int:
    return sum(1 for x in state[:-1] if x == L)


# ---------------------------------------------------------------------------
# Core DP
# ---------------------------------------------------------------------------

def count_hamiltonian_paths(
    n: int,
    order: list = None,
    checkpoint_dir: str = None,
    verbose: bool = False,
) -> int:
    """
    Count undirected Hamiltonian paths in G_n.

    Parameters
    ----------
    n              : graph size
    order          : vertex ordering (list of length n).  Auto-computed if None.
    checkpoint_dir : directory for save/resume checkpointing.
    verbose        : print per-step progress.

    Returns
    -------
    Exact count (int, arbitrary precision).
    """
    if n <= 1:
        return 0

    adj = build_graph(n)

    if order is None:
        if verbose:
            print(f"  Computing ordering for n={n} …", flush=True)
        order = best_order(adj, n, method="bfs+sa")

    if verbose:
        mx, pr = frontier_stats(adj, order)
        print(f"  Ordering: max_fw={mx}, profile={pr}", flush=True)

    pos = {v: i for i, v in enumerate(order)}
    last_step = {
        v: max((pos[w] for w in adj[v]), default=pos[v])
        for v in range(1, n + 1)
    }

    frontier = []
    frontier_set = set()
    dp: dict = defaultdict(int)
    dp[(0,)] = 1  # empty connectivity + n_closed=0

    start_step = 0
    if checkpoint_dir:
        cp = os.path.join(checkpoint_dir, f"ham_dp_n{n}.pkl")
        if os.path.exists(cp):
            with open(cp, "rb") as f:
                saved = pickle.load(f)
            frontier = saved["frontier"]
            frontier_set = set(frontier)
            dp = saved["dp"]
            start_step = saved["step"] + 1
            if verbose:
                print(f"  Resumed from step {start_step}", flush=True)

    total_paths = 0

    for step in range(start_step, n):
        v = order[step]

        # --- A. Introduce v -------------------------------------------
        frontier.append(v)
        frontier_set.add(v)
        new_dp: dict = defaultdict(int)
        for state, cnt in dp.items():
            new_dp[state[:-1] + (0,) + (state[-1],)] += cnt
        dp = new_dp
        v_idx = len(frontier) - 1

        # --- B. Edge decisions ----------------------------------------
        for w in sorted(adj[v]):
            if w not in frontier_set or pos[w] >= pos[v]:
                continue
            w_idx = frontier.index(w)
            new_dp = defaultdict(int)

            for state, cnt in dp.items():
                conn = state[:-1]
                nc = state[-1]
                sv = conn[v_idx]
                sw = conn[w_idx]

                # Exclude edge
                new_dp[state] += cnt

                # Include edge
                if sv == -1 or sw == -1:
                    continue

                ns = list(conn)

                if sv == 0 and sw == 0:
                    L = _max_label(state) + 1
                    ns[v_idx] = L; ns[w_idx] = L
                    new_dp[_canonicalize(tuple(ns) + (nc,))] += cnt

                elif sv == 0:
                    ns[w_idx] = -1; ns[v_idx] = sw
                    new_dp[_canonicalize(tuple(ns) + (nc,))] += cnt

                elif sw == 0:
                    ns[v_idx] = -1; ns[w_idx] = sv
                    new_dp[_canonicalize(tuple(ns) + (nc,))] += cnt

                elif sv == sw:
                    continue  # would close a cycle

                else:
                    sv_count = _count_label(state, sv)
                    sw_count = _count_label(state, sw)
                    ns[v_idx] = -1; ns[w_idx] = -1
                    for k in range(len(ns)):
                        if ns[k] == sw:
                            ns[k] = sv

                    if sv_count == 1 and sw_count == 1:
                        # Both chains are half-external → merge closes them
                        new_nc = nc + 1
                        if new_nc > 1:
                            continue  # two complete chains → invalid
                        if not all(x == -1 for x in ns):
                            continue  # other open entries remain → can't complete
                        new_dp[_canonicalize(tuple(ns) + (new_nc,))] += cnt
                    else:
                        new_dp[_canonicalize(tuple(ns) + (nc,))] += cnt

            dp = new_dp

        # --- C. Eliminate vertices ------------------------------------
        for u in [u for u in frontier if last_step[u] <= step]:
            u_idx = frontier.index(u)
            new_dp = defaultdict(int)

            for state, cnt in dp.items():
                conn = state[:-1]
                nc = state[-1]
                su = conn[u_idx]

                if su == 0:
                    continue  # unvisited → discard
                elif su == -1:
                    ns = conn[:u_idx] + conn[u_idx + 1:]
                    new_dp[_canonicalize(tuple(ns) + (nc,))] += cnt
                else:
                    remaining = conn[:u_idx] + conn[u_idx + 1:]
                    if su in remaining:
                        new_dp[_canonicalize(remaining + (nc,))] += cnt
                    else:
                        # Both endpoints external → chain complete
                        new_nc = nc + 1
                        if new_nc > 1:
                            continue
                        if not all(x == -1 for x in remaining):
                            continue
                        if step == n - 1:
                            total_paths += cnt
                        # else: isolated chain at intermediate step → discard

            frontier.remove(u)
            frontier_set.discard(u)
            dp = new_dp

        # Collect completions from the merge path (nc=1, all-interior frontier)
        if step == n - 1:
            total_paths += dp.get((1,), 0)

        if verbose:
            print(
                f"  step={step:3d}  v={v:3d}  fw={len(frontier):2d}"
                f"  #states={len(dp):9d}  total_so_far={total_paths}",
                flush=True,
            )

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            cp = os.path.join(checkpoint_dir, f"ham_dp_n{n}.pkl")
            with open(cp, "wb") as f:
                pickle.dump({"frontier": list(frontier), "dp": dict(dp),
                             "step": step}, f)

    return total_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    start_n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    end_n = int(sys.argv[2]) if len(sys.argv) > 2 else start_n
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    ckpt = None
    for arg in sys.argv:
        if arg.startswith("--checkpoint="):
            ckpt = arg.split("=", 1)[1]

    for n in range(start_n, end_n + 1):
        t0 = time.time()
        count = count_hamiltonian_paths(n, verbose=verbose, checkpoint_dir=ckpt)
        elapsed = time.time() - t0
        print(f"n={n:3d}  ham_paths={count}  ({elapsed:.2f}s)", flush=True)
