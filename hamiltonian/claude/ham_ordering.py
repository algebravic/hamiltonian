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
import random
import copy


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

def _dp_cost(adj: dict, order: list, pw_bound: int) -> float:
    """
    Estimate DP cost given a vertex ordering, without running the DP.

    Models the actual memory access pattern:

      running_cap  tracks the table capacity (slots) that the C code will
                   actually use, mirroring ht_ensure_clear's resize logic.
                   When states_out < states_in the table shrinks; when it
                   doesn't, the large table carries forward and the next
                   step scans it at full size.

      step cost  = running_cap × 2^n_back
                   (table slots to scan) × (subsets per source state)

    This correctly penalises a long plateau of net-expanding steps at peak
    fw, where the table stays large and every step pays the full scan cost.
    It also penalises n_back=2 at 20M states more than n_back=3 at 5M states,
    matching empirical ns/state observations.

    A small profile term encourages earlier eliminations as a tiebreaker.

    Returns None if the ordering exceeds pw_bound.
    """
    n = len(order)
    pos   = {v: i for i, v in enumerate(order)}
    # last_step[v]: last step at which v is still in frontier
    last_step = {v: max((pos[w] for w in adj[v]), default=pos[v])
                 for v in range(1, n + 1)}

    def _next_pow2(x):
        if x < 1: return 1
        x -= 1
        x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32
        return x + 1

    # Simulate state count progression (rough estimate using connectivity theory)
    # We track running_cap: the C table capacity after each step's ht_ensure_clear.
    # ht_ensure_clear sizes nxt to next_pow2(estimated_output / 0.75).
    # The estimate for fused_sweep is 1.5 × states_in.
    # If actual output > 1.5 × states_in × 0.75, resize fires → cap doubles.
    # But we don't know the actual output without running the DP.
    # Approximation: use the profile-observed expansion factor doesn't work for new orderings.
    # Instead use the theoretical max: output ≤ 2^n_back × states_in (loose upper bound).
    # Better: use 1.0 as a neutral estimate, but penalise scan cost regardless.
    #
    # Simpler and empirically validated: use states_in as a proxy for table size
    # (since cap ≈ next_pow2(states_in / 0.75) ≈ 2 × states_in typically),
    # and multiply by 2^n_back for subset cost.

    # Simulate frontier sojourn costs.
    # Key metrics:
    # 1. pressure × 2^n_back × (fw/pw)^2: models table-scan cost (unchanged)
    # 2. sojourn penalty: vertex that stays in frontier for many steps
    #    at peak fw contributes O(sojourn × Bell(pw)) to cumulative cost.
    #    Penalise long sojourns at near-peak frontier widths.
    # 3. profile (Σ fw): tiebreaker — with higher weight than before.
    #
    # The sojourn of vertex v = last_step[v] - pos[v].
    # Sojourn cost: sojourn[v] × (fw_at_introduction / pw)^2.
    # This directly penalises placing v early when it has late last-neighbors.

    # Geometric state-count proxy.  Multiplicative accumulation correctly
    # captures compound growth across consecutive expanding steps, which
    # additive/pressure-based tracking underestimates.  Long sojourns are
    # penalised implicitly: a vertex in the frontier for k steps participates
    # in k expand cycles, compounding the proxy geometrically.
    proxy   = 1.0
    cost    = 0.0
    profile = 0

    for step in range(n):
        v = order[step]
        fw = sum(1 for u in order[:step + 1]
                 if any(pos[nb] > step for nb in adj[u]))
        if fw > pw_bound:
            return None

        n_back = sum(1 for w in adj[v]
                     if pos[w] < step and w in set(order[:step]))
        n_elim = sum(1 for u in order[:step + 1]
                     if last_step[u] == step)

        fw_weight = (fw / pw_bound) ** 2
        expand    = float(1 << min(n_back, 20)) * fw_weight
        compress  = max(0.5 ** n_elim, 0.01)
        proxy     = max(proxy * expand * compress, 1.0)

        cost    += proxy
        profile += fw

    return cost + 0.001 * profile  # increased profile weight from 0.001


def sa_refine_order(
    adj: dict,
    n: int,
    init_order: list,
    pw_bound: int,
    n_iter: int = 100_000,
    seed: int = 42,
) -> list:
    """
    Refine *init_order* using simulated annealing while keeping
    max_fw <= pw_bound (preserving the exact pathwidth).

    Minimises the estimated DP cost:
      Σ_{peak-fw steps} 2^{n_back}  +  0.01 * profile

    This directly penalises back-edges at maximum-frontier steps,
    which drive the multi-edge subset enumeration cost.

    Parameters
    ----------
    init_order : starting ordering (typically the MaxSAT pathwidth order)
    pw_bound   : maximum allowed frontier width (= true pathwidth)
    n_iter     : number of SA iterations
    seed       : random seed
    """
    import math

    rng = random.Random(seed)
    order = list(init_order)
    cost = _dp_cost(adj, order, pw_bound)
    if cost is None:
        raise ValueError("init_order already exceeds pw_bound")

    best_order = order[:]
    best_cost = cost
    T = cost * 0.05
    if T == 0:
        return order
    decay = (cost * 1e-4 / T) ** (1.0 / n_iter)

    for _ in range(n_iter):
        i, j = sorted(rng.sample(range(n), 2))
        order[i], order[j] = order[j], order[i]

        new_cost = _dp_cost(adj, order, pw_bound)
        if new_cost is None:                       # violates pathwidth
            order[i], order[j] = order[j], order[i]
            continue

        delta = new_cost - cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_order = order[:]
        else:
            order[i], order[j] = order[j], order[i]

        T *= decay

    return best_order



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
