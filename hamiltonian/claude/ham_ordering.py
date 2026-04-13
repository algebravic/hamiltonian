"""
ham_ordering.py
---------------
Vertex ordering strategies for minimizing the frontier width of G_n
(the square-sum graph), used to reduce ZDD / DP state space.

Public API
----------
build_graph(n)            -> dict[int, set[int]]
frontier_width(adj, order) -> list[int]        (width at each step)
frontier_stats(adj, order) -> (max_fw, profile)
best_bfs_order(adj, n)    -> list[int]
min_fill_order(adj, n)    -> list[int]
sa_order(adj, n, init_order, ...) -> list[int]
best_order(adj, n, method="bfs+sa") -> list[int]
"""

from math import isqrt
import math
import random
import copy

from .sa_cost import sa_cost


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(n: int) -> dict:
    """Return adjacency dict for G_n (vertices 1..n, edge iff sum is a square)."""
    adj = {v: set() for v in range(1, n + 1)}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = i + j
            r = isqrt(s)
            if r * r == s:
                adj[i].add(j)
                adj[j].add(i)
    return adj


# ---------------------------------------------------------------------------
# Frontier width metrics
# ---------------------------------------------------------------------------

def frontier_width(adj: dict, order: list) -> list:
    """
    Return a list fw where fw[i] = number of frontier vertices just AFTER
    vertex order[i] has been introduced.

    A vertex u is in the frontier at step i if:
      - u has been introduced (pos[u] <= i)
      - u has at least one neighbor w with pos[w] > i  (still has future edges)
    """
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    fw = []
    for i in range(n):
        count = sum(
            1 for u in order[: i + 1]
            if any(pos[nb] > i for nb in adj.get(u, []))
        )
        fw.append(count)
    return fw


def frontier_stats(adj: dict, order: list) -> tuple:
    """Return (max_frontier_width, total_profile) for the given order."""
    fw = frontier_width(adj, order)
    return max(fw) if fw else 0, sum(fw)


# ---------------------------------------------------------------------------
# Ordering 1: Best BFS (O(n^2 * E))
# ---------------------------------------------------------------------------

def best_bfs_order(adj: dict, n: int) -> list:
    """
    Try BFS from every vertex as a start; return the ordering with the
    smallest (max_fw, profile) pair.

    BFS naturally keeps neighbouring vertices together, producing compact
    frontiers for sparse graphs.
    """
    vertices = list(range(1, n + 1))
    best = (10 ** 9, 10 ** 9, None)

    for start in vertices:
        order = _bfs_order(adj, start, vertices)
        mx, pr = frontier_stats(adj, order)
        if (mx, pr) < best[:2]:
            best = (mx, pr, order)

    return best[2]


def _bfs_order(adj: dict, start: int, all_vertices: list) -> list:
    visited = [False] * (max(all_vertices) + 1)
    queue = [start]
    visited[start] = True
    order = []
    head = 0
    while head < len(queue):
        v = queue[head]; head += 1
        order.append(v)
        for w in sorted(adj.get(v, [])):   # deterministic
            if not visited[w]:
                visited[w] = True
                queue.append(w)
    # Append any disconnected vertices
    order += sorted(v for v in all_vertices if not visited[v])
    return order


# ---------------------------------------------------------------------------
# Ordering 2: Greedy min-fill (treewidth heuristic)
# ---------------------------------------------------------------------------

def min_fill_order(adj: dict, n: int) -> list:
    """
    Greedy minimum-fill-in elimination order.  Repeatedly removes the vertex
    whose elimination requires fewest new edges (fill edges) among its
    neighbours.  Good for treewidth; mediocre for pathwidth but sometimes
    outperforms BFS on particular instances.
    """
    # Work on a mutable copy
    H = {v: set(nbrs) for v, nbrs in adj.items()}
    order = []
    remaining = set(range(1, n + 1))

    while remaining:
        best_v, best_fill = None, 10 ** 9
        for v in remaining:
            nbrs = list(H[v] & remaining)
            fill = sum(
                1 for i in range(len(nbrs))
                for j in range(i + 1, len(nbrs))
                if nbrs[j] not in H[nbrs[i]]
            )
            if fill < best_fill or (fill == best_fill and v < best_v):
                best_fill, best_v = fill, v

        order.append(best_v)
        # Add fill edges
        nbrs = list(H[best_v] & remaining)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if nbrs[j] not in H[nbrs[i]]:
                    H[nbrs[i]].add(nbrs[j])
                    H[nbrs[j]].add(nbrs[i])
        remaining.remove(best_v)

    return order


# ---------------------------------------------------------------------------
# Ordering 3: Simulated Annealing refinement
# ---------------------------------------------------------------------------

def sa_order(
    adj: dict,
    n: int,
    init_order: list,
    n_iter: int = 200_000,
    T_start: float = 3.0,
    T_end: float = 0.05,
    seed: int = 42,
) -> list:
    """
    Refine *init_order* using simulated annealing to minimise
    (max_frontier_width, profile).  Uses random swap and segment-reverse
    moves.

    Parameters
    ----------
    n_iter   : number of SA iterations (increase for larger n)
    T_start  : initial temperature
    T_end    : final temperature
    seed     : random seed for reproducibility
    """
    import math
    rng = random.Random(seed)

    order = list(init_order)
    fw = frontier_width(adj, order)
    best_order = order[:]
    best_max = max(fw)
    best_prof = sum(fw)

    decay = (T_end / T_start) ** (1.0 / n_iter)
    T = T_start

    for _ in range(n_iter):
        i, j = sorted(rng.sample(range(n), 2))

        # Choose move: 50% swap, 50% segment reverse
        if rng.random() < 0.5:
            order[i], order[j] = order[j], order[i]
        else:
            order[i : j + 1] = order[i : j + 1][::-1]

        fw_new = frontier_width(adj, order)
        new_max = max(fw_new)
        new_prof = sum(fw_new)

        cur_max = max(fw)
        cur_prof = sum(fw)
        # Primary objective: max_fw; secondary: profile
        delta = (new_max - cur_max) * 10.0 + 0.01 * (new_prof - cur_prof)

        if delta <= 0 or rng.random() < math.exp(-delta / T):
            fw = fw_new
            if new_max < best_max or (new_max == best_max and new_prof < best_prof):
                best_order = order[:]
                best_max = new_max
                best_prof = new_prof
        else:
            # Undo move
            if rng.random() < 0.5:  # was swap
                order[i], order[j] = order[j], order[i]
            else:
                order[i : j + 1] = order[i : j + 1][::-1]

        T *= decay

    return best_order


# ---------------------------------------------------------------------------
# Combined: best_order
# ---------------------------------------------------------------------------

def best_order(adj: dict, n: int, method: str = "bfs+sa",
               sa_iters: int = 200_000, seed: int = 42) -> list:
    """
    Return a high-quality vertex order using the chosen *method*.

    method options:
      "bfs"        – best-BFS over all start vertices
      "minfill"    – greedy min-fill-in
      "bfs+sa"     – best-BFS followed by SA refinement  (default)
      "minfill+sa" – min-fill followed by SA refinement
    """
    if "bfs" in method:
        init = best_bfs_order(adj, n)
    else:
        init = min_fill_order(adj, n)

    if "sa" in method:
        result = sa_order(adj, n, init, n_iter=sa_iters, seed=seed)
    else:
        result = init

    return result


# ---------------------------------------------------------------------------
# Secondary optimisation: refine a pathwidth-optimal ordering
# ---------------------------------------------------------------------------

def _dp_cost(adj: dict, order: list, pw_bound: int,
             expand_base: float = 1.55,
             density_alpha: float = 0.25,
             spike_penalty: float = 0.0) -> float:
    """
    Estimate DP cost given a vertex ordering, without running the DP.

    Cost function (V7): V6 with connected-component Bell correction.

    Key insight (Victor Miller): if G[F_i] has k connected components
    C_1,...,C_k, the valid state count is bounded by ∏_j Bell(|C_j|),
    not Bell(fw).  A fw=13 frontier with 10 components [2,2,2,1,...,1]
    has ∏Bell=8 vs Bell(13)=27.6M — a 3.4M× overestimate.

    V7 keeps the V6 proxy structure (numerical stability) and applies a
    multiplicative component correction at each step:

        step_cost = proxy × c^nb × (fw/pw)^2 × (1+α·e_bag/fw)
                           × sqrt(∏Bell(|C_j|) / Bell(fw))

    The sqrt is used for numerical stability (avoids extreme compression
    of the proxy when the correction is very small).

    Back-edges to isolated components (|C_j|=1) contribute no expansion
    since Bell(1)=1 means they have exactly one connectivity state.

    Returns None if fw > pw_bound and spike_penalty == 0.
    """
    BELL_PRE = [1,1,2,5,15,52,203,877,4140,21147,115975,678570,
                4213597,27644437,190899322,1382958545]
    def bell_n(k):
        return BELL_PRE[min(k, len(BELL_PRE)-1)]

    n = len(order)
    pos       = {v: i for i, v in enumerate(order)}
    last_step = {v: max((pos[w] for w in adj[v]), default=pos[v])
                 for v in range(1, n + 1)}

    proxy   = 1.0
    cost    = 0.0
    profile = 0
    frontier = set()

    for step in range(n):
        v = order[step]

        # Incremental frontier update: introduce v, then eliminate any vertex
        # whose last future neighbour has now been placed.
        # u leaves the frontier after step last_step[u] (its last neighbour's step),
        # so it should be absent from step last_step[u]+1 onward — i.e. remove
        # when last_step[u] <= current step.
        fw_prev = len(frontier)
        frontier.add(v)
        frontier -= {u for u in list(frontier) if last_step[u] <= step}

        fw = len(frontier)
        delta_fs = fw - fw_prev

        if fw > pw_bound:
            if spike_penalty == 0.0:
                return None
            extra_penalty = spike_penalty ** (fw - pw_bound)
        else:
            extra_penalty = 1.0

        # Connected components of G[frontier]
        vset = frontier
        visited = set()
        comp_of   = {}
        comp_sizes = []
        for start in vset:
            if start in visited:
                continue
            comp = set(); q = [start]
            while q:
                u = q.pop()
                if u in comp: continue
                comp.add(u)
                for w in adj[u]:
                    if w in vset and w not in comp: q.append(w)
            ci = len(comp_sizes)
            comp_sizes.append(len(comp))
            for u in comp: comp_of[u] = ci
            visited |= comp

        # Component correction: sqrt(∏Bell(|C_j|) / Bell(fw))
        # Applied to fw_weight to discount fragmented frontiers.
        # Note: G[frontier] graph components overestimate DP state reduction
        # (path connectivity can link components through non-frontier vertices)
        # so sqrt rather than linear is used as a partial correction.
        bell_prod = math.prod(bell_n(s) for s in comp_sizes)
        bell_fw   = bell_n(fw)
        log_correction = 0.5 * (math.log(max(bell_prod, 1)) -
                                 math.log(max(bell_fw, 1)))
        comp_correction = math.exp(log_correction)   # in (0, 1]

        # fw_weight with component correction
        fw_weight = (fw / pw_bound) ** 2 * extra_penalty * comp_correction

        # nb: ALL back-edges contribute (path connectivity through already-
        # placed vertices means even graph-isolated components can be linked)
        n_back = sum(1 for w in adj[v] if pos[w] < step and w in frontier)

        e_bag = sum(1 for u in frontier for w in adj[u]
                    if w in frontier and pos[u] < pos[w])

        # Fitted model replaces the old c^n_back proxy.
        # sa_cost(fs, delta_fs, n_back, e_bag) predicts log(so/si); exp() converts
        # to a multiplicative expansion factor.
        # density_amp is dropped: e_bag is now a direct model feature.
        # fw_weight is retained for its spike_penalty and Bell comp_correction terms,
        # which are structural constraints not captured by the regression.
        expand  = math.exp(sa_cost(fw, delta_fs, n_back, e_bag)) * fw_weight
        n_elim  = sum(1 for u in order[:step + 1] if last_step[u] == step)
        compress = max(0.5 ** n_elim, 0.01)

        cost  += proxy * expand
        proxy  = max(proxy * expand * compress, 1.0)
        profile += fw

    return cost + 0.001 * profile


def sa_refine_order(
    adj: dict,
    n: int,
    init_order: list,
    pw_bound: int,
    n_iter: int = 100_000,
    seed: int = 42,
    expand_base: float = 1.55,
    density_alpha: float = 0.25,
    spike_penalty: float = 0.0,
) -> list:
    """
    Refine *init_order* using simulated annealing.

    By default (spike_penalty=0.0) the exact pathwidth is preserved: any
    swap that raises max_fw above pw_bound is rejected.

    With spike_penalty > 0, the constraint is relaxed to a soft penalty:
    over-budget steps are penalised by spike_penalty^overage in the cost
    function but not hard-rejected.  This lets SA trade a small number of
    early spikes (cheap, because the proxy state count is low) for a
    smoother overall profile.  spike_penalty=16 is a reasonable starting
    point: fw=pw+1 costs 16x the normal rate at that proxy level.

    The returned ordering may have max_fw > pw_bound when spike_penalty > 0.
    Use frontier_stats() to check the actual max after refinement.
    """
    import math

    rng = random.Random(seed)
    order = list(init_order)
    cost = _dp_cost(adj, order, pw_bound, expand_base=expand_base,
                    density_alpha=density_alpha, spike_penalty=spike_penalty)
    if cost is None:
        raise ValueError("init_order already exceeds pw_bound")

    best_order = order[:]
    best_cost = cost
    best_iter = 0
    T = cost * 0.05
    if T == 0:
        return best_order
    decay = (cost * 1e-4 / T) ** (1.0 / n_iter)

    for it in range(1, n_iter + 1):
        i, j = sorted(rng.sample(range(n), 2))
        order[i], order[j] = order[j], order[i]

        new_cost = _dp_cost(adj, order, pw_bound, expand_base=expand_base,
                            density_alpha=density_alpha,
                            spike_penalty=spike_penalty)
        if new_cost is None:
            order[i], order[j] = order[j], order[i]
            T *= decay
            continue

        delta = new_cost - cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_order = order[:]
                best_iter = it
        else:
            order[i], order[j] = order[j], order[i]

        T *= decay

    return best_order, best_iter, best_cost



def best_multistart_order(
    adj: dict,
    n: int,
    G,
    pw_bound: int,
    n_iter: int = 100_000,
    max_solutions: int = 50,
    fast_threshold_s: float = 0.5,
    consecutive_fast_stop: int = 10,
    expand_base: float = 1.55,
    density_alpha: float = 0.25,
    spike_penalty: float = 0.0,
    stratified: bool = False,
    solver: str = "cd195",
    verbose: bool = True,
) -> tuple:
    """
    Run SA refinement from every topologically distinct optimal-pw ordering
    produced by the all_solutions generator.

    Stopping criterion: generator inter-arrival time.  Solutions produced in
    < fast_threshold_s seconds are cheap local variations reachable by SA
    from any other start.  Stop after consecutive_fast_stop such solutions.

    Returns
    -------
    (best_order, best_cost, all_candidates, n_distinct, n_total)

    where all_candidates is a list of (sa_cost, order) for every solution
    tried, sorted by sa_cost ascending.  Pass this to validate_multistart_orders
    to re-rank by actual partial-DP time.
    """
    import time
    try:
        from separation.vertex_separation import pathwidth_order
    except ImportError:
        raise ImportError(
            "The 'separation' package is not installed.\n"
            "Clone https://github.com/algebravic/separation and pip install -e ."
        )

    best_order_out = None
    best_cost = float('inf')
    all_candidates = []   # (sa_cost, order) for every tried solution
    n_distinct = 0
    n_total = 0
    consecutive_fast = 0

    if verbose:
        print(f"  Multi-start SA: pw={pw_bound}, SA iters={n_iter:,} per start",
              flush=True)
        if stratified:
            print(f"  stratified=True, solver={solver}", flush=True)
        else:
            print(f"  unstratified, solver={solver}", flush=True)

    gen = pathwidth_order(G, bound=pw_bound, all_solutions=True,
                          stratified=stratified, solver=solver)
    t_last = time.time()

    for pw, order in gen:
        t_gen = time.time()
        gen_time = t_gen - t_last

        n_total += 1

        is_new_skeleton = (n_total == 1) or (gen_time >= fast_threshold_s)
        if is_new_skeleton:
            n_distinct += 1
            consecutive_fast = 0
        else:
            consecutive_fast += 1

        refined, best_iter, sa_best_cost = sa_refine_order(
                                  adj, n, order, pw_bound,
                                  n_iter=n_iter, seed=n_total * 17 + 42,
                                  expand_base=expand_base,
                                  density_alpha=density_alpha,
                                  spike_penalty=spike_penalty)
        cost = sa_best_cost   # already computed inside sa_refine_order
        all_candidates.append((cost, refined[:]))

        is_best = (cost is not None and cost < best_cost)
        if is_best:
            best_cost = cost
            best_order_out = refined[:]

        if verbose:
            marker = "***" if is_best else "   "
            sk = "  [new skeleton]" if is_new_skeleton else f"  [fast {gen_time:.3f}s]"
            pct = 100 * best_iter / n_iter
            print(f"  {marker} sol {n_total:3d}: SA cost={cost:.3e}  "
                  f"best={best_cost:.3e}  gen={gen_time:.2f}s  "
                  f"best@{best_iter:,}/{n_iter:,} ({pct:.0f}%){sk}", flush=True)

        if n_total > max_solutions:
            if verbose:
                print(f"  Reached max_solutions={max_solutions}, stopping.", flush=True)
            break

        if consecutive_fast >= consecutive_fast_stop:
            if verbose:
                print(f"  {consecutive_fast} consecutive fast solutions — "
                      f"all {n_distinct} distinct skeletons found, stopping.", flush=True)
            break

        t_last = time.time()

    all_candidates.sort(key=lambda x: x[0])   # sort by SA cost ascending

    if verbose:
        print(f"  Multi-start done: {n_total} solutions tried, "
              f"{n_distinct} distinct skeletons, best cost={best_cost:.3e}", flush=True)

    return best_order_out, best_cost, all_candidates, n_distinct, n_total


def validate_multistart_orders(
    candidates: list,
    adj: dict,
    n: int,
    step_limit: int,
    verbose: bool = True,
) -> list:
    """
    Rank candidate orderings by actual partial-DP wall-clock time, running
    only the first *step_limit* steps of the frontier DP.

    Corrects proxy miscalibration: orderings that score well on _dp_cost but
    are slower in practice (e.g. due to bad tail behaviour) are re-ranked by
    measured early-phase cost.  step_limit = n // 2 typically captures the
    expensive peak steps while running in a fraction of the full DP time.

    Parameters
    ----------
    candidates  : list of (sa_cost, order) tuples from best_multistart_order
    adj         : adjacency dict
    n           : number of vertices
    step_limit  : DP steps to run per candidate (n//2 is a good default)
    verbose     : print per-candidate timing

    Returns
    -------
    List of (partial_ms, sa_cost, order) sorted by partial_ms ascending.
    """
    from .ham_dp_c import partial_dp_time_c

    if verbose:
        print(f"  Validating {len(candidates)} candidates: "
              f"first {step_limit}/{n} DP steps each", flush=True)

    results = []
    for rank, (sa_cost, order) in enumerate(candidates):
        ms = partial_dp_time_c(n, order, adj, step_limit)
        results.append((ms, sa_cost, order))
        if verbose:
            marker = ""
            if results and ms < results[0][0]:
                marker = "  ***"
            print(f"    candidate {rank+1:3d}: partial_dp={ms:8.1f}ms  "
                  f"sa_cost={sa_cost:.3e}{marker}", flush=True)

    results.sort(key=lambda x: x[0])
    if verbose:
        print(f"  Best by partial DP (step 0..{step_limit-1}): "
              f"{results[0][0]:.1f}ms  (sa_cost={results[0][1]:.3e})",
              flush=True)
    return results


def graphillion_edge_universe(adj: dict, n: int, order: list) -> list:
    """
    Return the list of edges sorted in the order that graphillion should use
    as its 'universe'.  Edges are sorted by (max_pos, min_pos) where pos is
    the position in *order*.  This aligns the ZDD variable order with the
    frontier DP interpretation and typically reduces ZDD size.
    """
    pos = {v: i for i, v in enumerate(order)}
    edges = [
        (i, j)
        for i in range(1, n + 1)
        for j in range(i + 1, n + 1)
        if isqrt(i + j) ** 2 == i + j
    ]
    edges.sort(key=lambda e: (max(pos[e[0]], pos[e[1]]), min(pos[e[0]], pos[e[1]])))
    return edges


# ---------------------------------------------------------------------------
# CLI: print ordering info
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 56
    method = sys.argv[2] if len(sys.argv) > 2 else "bfs+sa"

    print(f"Building G_{n} ...", flush=True)
    adj = build_graph(n)

    print(f"Computing ordering (method={method}) ...", flush=True)
    order = best_order(adj, n, method=method)
    mx, pr = frontier_stats(adj, order)

    print(f"n={n}: max_frontier_width={mx}, profile={pr}")
    print(f"Order: {order}")
