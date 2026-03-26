"""
ham_count.py
------------
Exact Hamiltonian path counter.

Two modes of operation
----------------------
1. Square-sum graph G_n  (default)
   Vertices {1,...,n}, edge (i,j) iff i+j is a perfect square.

       python ham_count.py [start_n] [end_n] [options]

2. Arbitrary graph from a DIMACS file  (--graph option)
   Standard DIMACS edge-list format:
       c  comment lines (ignored)
       p  edge <n_vertices> <n_edges>
       e  <u> <v>   (one per edge, 1-indexed)

       python ham_count.py --graph mygraph.col [options]

Pipeline
--------
1. Build/read the graph.
2. Compute the optimal pathwidth vertex ordering via the `separation`
   package (algebravic/separation on GitHub), which encodes the problem
   as MaxSAT and solves it with RC2.
3. Run the C frontier DP (ham_dp_c.py) with that ordering.

Options
-------
  --graph FILE     Read graph from a DIMACS file instead of building G_n.
  --no-pw          Use best-BFS ordering instead of MaxSAT pathwidth.
  --no-refine      Skip the SA refinement step.
  --refine-iters N SA iterations (default: 100000).
  --bound K        Pathwidth upper bound hint for the MaxSAT solver.
  --stratified     Use RC2Stratified (sometimes faster for large n).
  --solver NAME    SAT solver backend for RC2 (default: cd195).
  --checkpoint P   Path for checkpoint file.
  --checkpoint-interval SECS  Checkpoint interval (default: 300s).
  -v / --verbose   Print per-step frontier DP progress to stderr.
  --profile        Alias for --verbose.

Dependencies
------------
    pip install python-sat networkx cffi
    git clone https://github.com/algebravic/separation
    pip install -e separation/           # or add src/ to PYTHONPATH
"""

import sys
import time
import argparse
import gc
import os
import tempfile
from math import isqrt

import networkx as nx

from .ham_dp_c import count_hamiltonian_paths_c, count_hamiltonian_paths_sm,  _get_lib
from .ham_ordering import build_graph, best_bfs_order, frontier_stats, sa_refine_order, best_multistart_order, validate_multistart_orders


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_gnx(n: int) -> nx.Graph:
    """Build G_n as a networkx Graph (vertices 1..n)."""
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = i + j
            r = isqrt(s)
            if r * r == s:
                G.add_edge(i, j)
    return G


def read_dimacs(path: str) -> nx.Graph:
    """
    Read a graph in DIMACS edge-list format.

    Format:
        c  <comment>         (ignored)
        p  edge <n> <m>      (declares n vertices, m edges)
        e  <u> <v>           (one undirected edge per line, 1-indexed)

    Vertices are renumbered to 1..n in the order they first appear if the
    file uses non-contiguous labels, but contiguous 1-indexed files are
    handled without renumbering.

    Returns
    -------
    nx.Graph with integer vertex labels starting at 1.
    """
    G = nx.Graph()
    n_declared = None
    seen = {}   # original label → internal 1-indexed label
    next_id = [1]

    def intern(v):
        if v not in seen:
            seen[v] = next_id[0]
            next_id[0] += 1
        return seen[v]

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('c') or line.startswith('%'):
                continue
            parts = line.split()
            if parts[0] == 'p':
                # p edge <n> <m>  OR  p tw <n> <m>  (htd uses 'tw')
                n_declared = int(parts[2])
                G.add_nodes_from(range(1, n_declared + 1))
            elif parts[0] in ('e', 'a'):
                u, v = int(parts[1]), int(parts[2])
                if n_declared is not None:
                    # contiguous 1-indexed: use labels directly
                    G.add_edge(u, v)
                else:
                    # no header: intern labels
                    G.add_edge(intern(u), intern(v))

    if G.number_of_nodes() == 0:
        raise ValueError(f"No vertices found in DIMACS file: {path}")

    return G


def nx_to_adj(G: nx.Graph) -> dict:
    """Convert a networkx Graph to the adjacency-set dict expected by the C DP."""
    return {v: set(G.neighbors(v)) for v in G.nodes()}


# ---------------------------------------------------------------------------
# Optimal ordering via the separation package
# ---------------------------------------------------------------------------

def get_pathwidth_order(
    G: nx.Graph,
    bound: int = None,
    stratified: bool = False,
    solver: str = "cd195",
    minimize_profile: bool = False,
) -> tuple:
    """
    Compute the exact pathwidth and an optimal vertex ordering for G
    using the MaxSAT-based solver from algebravic/separation.

    If minimize_profile=True, a two-phase solve is performed:
      1. Find pw* (the exact pathwidth) via primary MaxSAT minimisation.
      2. Re-solve with bound=pw* and minimize_profile=True to find the
         ordering that minimises Σ_t |F_t| (profile) among all orderings
         achieving pw*.  This uses soft clauses on the u[v,t] variables
         which already encode frontier membership.

    Returns (pathwidth : int, order : list of vertices).
    """
    try:
        from separation.vertex_separation import pathwidth_order
    except ImportError:
        raise ImportError(
            "The 'separation' package is not installed.\n"
            "Clone https://github.com/algebravic/separation\n"
            "and install it (pip install -e separation/),\n"
            "or pass --no-pw to use the BFS heuristic instead."
        )

    # Phase 1: find optimal pathwidth (always needed)
    pw, order = pathwidth_order(
        G,
        bound=bound,
        solver=solver,
        stratified=stratified,
    )

    # Phase 2 (optional): re-solve to minimise profile at fixed pw
    if minimize_profile:
        print(f"  minimize_profile: re-solving with bound={pw} to minimise Σ|F_t|",
              flush=True)
        pw2, order = pathwidth_order(
            G,
            bound=pw,
            minimize_profile=True,
            solver=solver,
            stratified=stratified,
        )
        assert pw2 == pw, f"Profile minimisation changed pw: {pw2} != {pw}"

    return pw, order


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Count Hamiltonian paths in a graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # --- Input source (mutually exclusive) ---
    src = p.add_mutually_exclusive_group()
    src.add_argument("--graph", metavar="FILE",
                     help="DIMACS edge-list file.  Overrides start_n/end_n.")
    # Positional args for the G_n mode
    p.add_argument("start_n", type=int, nargs="?", default=15,
                   help="First n to compute in G_n mode (default: 15).")
    p.add_argument("end_n",   type=int, nargs="?", default=None,
                   help="Last n to compute in G_n mode (default: same as start_n).")

    # --- Ordering options ---
    p.add_argument("--no-pw", action="store_true",
                   help="Use BFS ordering instead of MaxSAT pathwidth.")
    p.add_argument("--no-refine", action="store_true",
                   help="Skip the SA refinement of the MaxSAT ordering.")
    p.add_argument("--multi-start", action="store_true",
                   help="Run SA from all topologically distinct MaxSAT orderings "
                        "and pick the best.  Implies --refine.  Prints per-solution "
                        "cost so you can see whether extra starts help.")
    p.add_argument("--multi-start-max", type=int, default=50, metavar="N",
                   help="Max solutions to draw from the generator (default: 50).")
    p.add_argument("--multi-start-validate", type=int, default=0, metavar="K",
                   help="After multi-start SA, re-rank candidates by running the "
                        "first K DP steps and pick the fastest. K=0 disables "
                        "(default). Suggested: K = n//2.")
    p.add_argument("--refine-iters", type=int, default=100_000, metavar="N",
                   help="SA iterations for secondary refinement (default: 100000).")
    p.add_argument("--expand-base", type=float, default=1.55, metavar="C",
                   help="Expansion base c in proxy cost c^n_back (default: 1.55). "
                        "Calibrated from n=59/61 profile data; c=2.0 gives original "
                        "behaviour. Empirical optimum ~1.55 (error valley 1.4–1.7).")
    p.add_argument("--density-alpha", type=float, default=0.25, metavar="A",
                   help="Weight of intra-bag edge density in proxy cost (default: 0.25). "
                        "Adds (1 + A*e_bag/fw) as a density amplifier. "
                        "Set 0.0 to disable (V5 behaviour).")
    p.add_argument("--spike-penalty", type=float, default=0.0, metavar="P",
                   help="Soft-relax the pathwidth constraint in SA (default: 0.0 = hard). "
                        "Steps exceeding pw are penalised by P^overage rather than "
                        "rejected outright, letting SA trade early spikes for a smoother "
                        "profile.  P=16 is a reasonable starting point.  "
                        "The resulting ordering may exceed pw_bound.")
    p.add_argument("--bound", type=int, default=None, metavar="K",
                   help="Pathwidth upper bound hint for the MaxSAT solver.")
    p.add_argument("--stratified", action="store_true",
                   help="Use RC2Stratified (sometimes faster for large n).")
    p.add_argument("--minimize-profile", action="store_true",
                   help="After finding pw*, re-solve with a secondary MaxSAT objective "
                        "to minimise the profile (Σ_t |F_t|) among all pw*-optimal "
                        "orderings.  Uses soft clauses on the u[v,t] frontier variables. "
                        "Adds a second solver invocation but produces orderings with "
                        "smaller total frontier exposure.")
    p.add_argument("--solver", default="cd195", metavar="NAME",
                   help="SAT solver for RC2 (default: cd195).")

    # --- DP options ---
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print per-step frontier DP progress.")
    p.add_argument("--profile", action="store_true",
                   help="Print per-step timing/state-count table to stderr.")
    p.add_argument("--checkpoint", default="", metavar="PATH",
                   help="Path for checkpoint file.")
    p.add_argument("--checkpoint-interval", type=float, default=300.0, metavar="SECS",
                   help="Save a checkpoint at most every SECS seconds (default 300).")
    p.add_argument("--load-factor", type=int, default=75, metavar="PCT",
                   choices=[75, 80, 85, 90],
                   help="Hash-table load factor %% (75/80/85/90, default 75). "
                        "Higher values reduce memory usage at the cost of more "
                        "hash probes per insert.  85 is recommended on machines "
                        "where peak two-table memory exceeds available RAM.")
    p.add_argument("--sort-merge", action="store_true",
                   help="Use the sort/merge implementation.")
    return p.parse_args()


def _run_one(label, n_vertices, G, adj, args, use_pw):
    """
    Run the full ordering + DP pipeline for a single graph.

    Parameters
    ----------
    label      : string identifying this graph (for output)
    n_vertices : number of vertices
    G          : networkx Graph
    adj        : adjacency dict for the C DP
    args       : parsed CLI arguments
    use_pw     : whether to attempt MaxSAT pathwidth ordering

    Returns
    -------
    (use_pw, count) — use_pw may be set False if import failed.
    """
    # --- Vertex ordering ---
    t_ord = time.time()
    pw = None

    if use_pw:
        try:
            pw, order = get_pathwidth_order(
                G,
                bound=args.bound,
                stratified=args.stratified,
                solver=args.solver,
                minimize_profile=getattr(args, 'minimize_profile', False),
            )
        except ImportError as e:
            print(f"  WARNING: {e}\n  Falling back to BFS.", flush=True)
            use_pw = False

    if not use_pw or pw is None:
        order = best_bfs_order(adj, n_vertices)

    # --- SA refinement ---
    if pw is not None and not args.no_refine:
        eb = getattr(args, 'expand_base', 1.55)
        da = getattr(args, 'density_alpha', 0.25)
        sp = getattr(args, 'spike_penalty', 0.0)
        if getattr(args, 'multi_start', False):
            order, ms_cost, ms_candidates, ms_distinct, ms_total = best_multistart_order(
                adj, n_vertices, G, pw,
                n_iter=args.refine_iters,
                max_solutions=args.multi_start_max,
                expand_base=eb,
                density_alpha=da,
                spike_penalty=sp,
                stratified=args.stratified,
                solver=args.solver,
                verbose=True,
            )
            print(f"  multi-start: {ms_total} solutions, {ms_distinct} distinct, "
                  f"best SA cost={ms_cost:.3e}  "
                  f"(expand_base={eb}, density_alpha={da}, spike_penalty={sp})",
                  flush=True)

            # Optional: re-rank top candidates by actual partial-DP timing
            k = getattr(args, 'multi_start_validate', 0)
            if k > 0 and ms_candidates:
                top = ms_candidates[:min(10, len(ms_candidates))]
                ranked = validate_multistart_orders(
                    top, adj, n_vertices, step_limit=k, verbose=True)
                order = ranked[0][2]
                print(f"  validate: best by partial DP ({k} steps) "
                      f"has SA cost={ranked[0][1]:.3e}", flush=True)
        else:
            order, best_iter, _ = sa_refine_order(
                                    adj, n_vertices, order, pw,
                                    n_iter=args.refine_iters,
                                    expand_base=eb,
                                    density_alpha=da,
                                    spike_penalty=sp)
            if getattr(args, 'verbose', False) or True:
                pct = 100 * best_iter / args.refine_iters
                print(f"  SA best found at iter {best_iter:,}/{args.refine_iters:,} "
                      f"({pct:.0f}%)", flush=True)

    t_ord = time.time() - t_ord
    mx, pr = frontier_stats(adj, order)

    # --- Free solver memory before DP ---
    gc.collect()
    if pw is not None and n_vertices >= 50:
        time.sleep(0.5)

    # --- Checkpoint path ---
    ckpt_path = args.checkpoint
    if not ckpt_path and n_vertices >= 50:
        safe_label = label.replace('/', '_').replace(' ', '_')
        ckpt_path = os.path.join(tempfile.gettempdir(),
                                 f"ham_ckpt_{safe_label}.bin")

    # --- C frontier DP ---
    verbose = args.verbose or args.profile
    t_dp = time.time()

    if args.sort_merge:
        count = count_hamiltonian_paths_sm(
            n_vertices, order, adj,
            verbose=verbose,
            )
    else:
        count = count_hamiltonian_paths_c(
            n_vertices, order, adj,
            verbose=verbose,
            checkpoint_path=ckpt_path,
            checkpoint_secs=args.checkpoint_interval,
            load_factor=args.load_factor,
            )
    t_dp = time.time() - t_dp

    pw_str = f"pw={pw:2d}" if pw is not None else "pw= ?"
    spike_str = f" [SPIKE +{mx-pw}]" if (pw is not None and mx > pw) else ""
    print(
        f"{label}  {pw_str}  max_fw={mx:2d}  profile={pr:5d}"
        f"  ord={t_ord:.2f}s  dp={t_dp:.3f}s  ham_paths={count}{spike_str}",
        flush=True,
    )
    return use_pw, count


def main():
    args = parse_args()
    use_pw = not args.no_pw

    print("# Compiling C library …", flush=True)
    _get_lib()
    print("# Ready.", flush=True)

    if args.graph:
        # ---------------------------------------------------------------
        # DIMACS mode: single graph file
        # ---------------------------------------------------------------
        path = args.graph
        label = os.path.basename(path)
        print(f"# Graph: {path}  ordering={'MaxSAT-pathwidth' if use_pw else 'BFS-heuristic'}",
              flush=True)

        G = read_dimacs(path)
        adj = nx_to_adj(G)
        n = G.number_of_nodes()
        print(f"# {n} vertices, {G.number_of_edges()} edges", flush=True)

        _run_one(f"graph={label}", n, G, adj, args, use_pw)

    else:
        # ---------------------------------------------------------------
        # Square-sum G_n mode
        # ---------------------------------------------------------------
        start_n = args.start_n
        end_n   = args.end_n if args.end_n is not None else start_n

        print(f"# n = {start_n}..{end_n}  "
              f"ordering={'MaxSAT-pathwidth' if use_pw else 'BFS-heuristic'}",
              flush=True)

        for n in range(start_n, end_n + 1):
            adj = build_graph(n)
            G   = build_gnx(n)
            use_pw, _ = _run_one(f"n={n:3d}", n, G, adj, args, use_pw)


if __name__ == "__main__":
    main()
