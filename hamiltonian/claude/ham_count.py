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
2. Compute a pathwidth-optimal vertex ordering, by one of:
   --ordering kpw     the kobayashi-pw DAG of pathwidth-optimal orderings
                      (kpw/kpw2 -c), then the exact minimum of a secondary
                      cost over it (ham_kpw.py; default cost: the fitted
                      work model in work_model.py);
   --ordering maxsat  the `separation` package (algebravic/separation),
                      MaxSAT solved with RC2, then simulated annealing
                      (the default, and the method used for n <= 75);
   --ordering bfs     best BFS ordering (no optimality guarantee).
3. Run the C frontier DP (ham_dp_c.py) with that ordering.

Options
-------
  --graph FILE     Read graph from a DIMACS file instead of building G_n.
  --ordering M     maxsat (default), kpw, or bfs.
  --kpw-cost C     (kpw) work (default), v7 or v2.
  --no-pw          Alias for --ordering bfs.
  --no-refine      Skip the SA refinement step.
  --refine-iters N SA iterations (default: 100000).
  --bound K        Pathwidth upper bound hint for the MaxSAT solver.
  --stratified     Use RC2Stratified (sometimes faster for large n).
  --solver NAME    SAT solver backend for RC2 (default: cd195).
  --ord-verbose L  RC2 verbosity level during ordering (default: 0). Use 10
                   to see cost improvements as the MaxSAT solver progresses —
                   essential for diagnosing whether a long ordering phase is
                   converging or stuck.
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

from .ham_dp_c import count_hamiltonian_paths_c, count_hamiltonian_paths_sm, _get_lib, set_machine_yaml
from .ham_ordering import build_graph, best_bfs_order, frontier_stats, sa_refine_order, best_multistart_order, validate_multistart_orders
from . import ham_kpw


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
    profile_time_limit: float = 120.0,
    verbose: int = 0,
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
        verbose=verbose,
    )

    # Phase 2 (optional): re-solve to minimise profile at fixed pw
    if minimize_profile:
        import threading
        print(f"  minimize_profile: re-solving with bound={pw} to minimise Σ|F_t| "
              f"(time_limit={profile_time_limit}s)", flush=True)

        result_holder = [None]
        def _solve():
            try:
                r = pathwidth_order(G, bound=pw, minimize_profile=True,
                                    solver=solver, stratified=stratified,
                                    verbose=verbose)
                result_holder[0] = r
            except Exception as e:
                result_holder[0] = e

        t = threading.Thread(target=_solve, daemon=True)
        t.start()
        t.join(timeout=profile_time_limit)

        if result_holder[0] is None:
            print(f"  minimize_profile: timed out after {profile_time_limit}s, "
                  f"keeping phase-1 ordering", flush=True)
        elif isinstance(result_holder[0], Exception):
            print(f"  minimize_profile: error ({result_holder[0]}), "
                  f"keeping phase-1 ordering", flush=True)
        else:
            pw2, order = result_holder[0]
            _, prof_new = None, None
            print(f"  minimize_profile: done (pw={pw2})", flush=True)

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

    # --- Config file -------------------------------------------------------
    p.add_argument("--config", metavar="FILE",
                   help="YAML file whose keys match long option names (dashes → "
                        "underscores, leading dashes dropped).  Values are loaded "
                        "as defaults before CLI args are applied, so any flag "
                        "given on the command line overrides the file.  "
                        "Example: refine_iters: 500000  spike_penalty: 16")

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
    p.add_argument("--ordering", choices=["maxsat", "kpw", "bfs"], default="maxsat",
                   help="How to compute the vertex ordering: maxsat (MaxSAT pathwidth via "
                        "`separation`, then simulated annealing; default), kpw (exact "
                        "pathwidth DAG from kobayashi-pw, then the exact optimum of "
                        "--kpw-cost over it), or bfs (heuristic).")
    p.add_argument("--no-pw", action="store_true",
                   help="Alias for --ordering bfs.")
    p.add_argument("--kpw-cost", choices=list(ham_kpw.COSTS), default="work",
                   help="(kpw) Secondary cost minimised exactly over the DAG: work = "
                        "predicted DP work from the fitted model (default; falls back to "
                        "v7 if --work-model is missing), v7 = _dp_cost, v2 = additive.")
    p.add_argument("--work-model", default=None, metavar="PATH",
                   help="(kpw) Fitted work model JSON (default: work_model.json next to "
                        "work_model.py; make it with `fit-work-model`).")
    p.add_argument("--kpw-bin", default=None, metavar="PATH",
                   help="(kpw) kpw/kpw2 binary (default: kobayashi-pw's src/C, then PATH; "
                        "kpw2 is needed for n > 64).  Without one the DAG is built in Python.")
    p.add_argument("--kpw-threads", type=int, default=0, metavar="N",
                   help="(kpw) Threads for kpw2 (default 0 = all cores).")
    p.add_argument("--dag-cache", default=ham_kpw.DEFAULT_DAG_CACHE, metavar="DIR",
                   help="(kpw) Directory for cached DAG files (default: data/dags). "
                        "Use '' to disable caching.")
    p.add_argument("--kpw-greedy2", action="store_true",
                   help="(kpw) Second greedy rule: smaller DAG, usually worse orderings.")
    p.add_argument("--kpw-max-labels", type=int, default=None, metavar="L",
                   help="(kpw) Keep at most L Pareto labels per DAG node (bounds the "
                        "optimiser's memory; the result may then be suboptimal).")
    p.add_argument("--kpw-engine", choices=["kopt", "dsl"], default="kopt",
                   help="(kpw) kopt: DAG from kpw/kpw2 (cached) optimised by kopt (default); "
                        "dsl: kobayashi-pw's DSL (engine E2), same orderings, one process, no DAG file.")
    p.add_argument("--kpw-python", action="store_true",
                   help="(kpw) Optimise in Python even if the C optimiser kopt is built.")
    p.add_argument("--kopt-bin", default=None, metavar="PATH",
                   help="(kpw) kopt binary (default: kobayashi-pw's src/C, then PATH).")
    p.add_argument("--kpw-polish", action="store_true",
                   help="(kpw) Run the simulated annealing of --ordering maxsat from the "
                        "DAG optimum (off by default).")
    p.add_argument("--kpw-candidates", type=int, default=0, metavar="K",
                   help="(kpw) Also generate the K best V2 orderings and the optima of the "
                        "other costs, and choose among them by a partial DP over the first "
                        "--multi-start-validate steps (default n//2).  0 = off.")
    p.add_argument("--no-refine", action="store_true",
                   help="(maxsat) Skip the SA refinement of the MaxSAT ordering.")
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
    p.add_argument("--sa-t-start", type=float, default=0.0, metavar="T",
                   help="SA initial temperature (default: 0 → 5%% of initial cost). "
                        "Increase to explore more aggressively early on — useful "
                        "with --spike-penalty where the landscape is rough.")
    p.add_argument("--sa-t-end", type=float, default=0.0, metavar="T",
                   help="SA final temperature (default: 0 → 0.01%% of initial cost).")
    p.add_argument("--sa-schedule", default="geometric", metavar="S",
                   choices=["geometric", "linear", "cosine", "log"],
                   help="SA temperature schedule: geometric (default), linear, cosine, "
                        "or log (T=t0/log(1+kt), near-optimal per Stander & Silverman 1994).")
    p.add_argument("--sa-restarts", type=int, default=0, metavar="K",
                   help="Reheat SA K times after each full cool-down, resuming from "
                        "the best ordering found so far.  n_iter is split evenly across "
                        "all rounds.  Default: 0 (no restarts).  Useful with "
                        "--spike-penalty to escape early bad basins.")
    p.add_argument("--sa-restart-factor", type=float, default=0.5, metavar="F",
                   help="Temperature multiplier at each restart (default: 0.5 → each "
                        "restart starts at half the previous initial temperature).")
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
    p.add_argument("--profile-time-limit", type=float, default=120.0, metavar="SECS",
                   help="Time limit in seconds for the profile-minimisation phase 2 "
                        "solve (default: 120). If exceeded, the phase-1 ordering is "
                        "kept. Set higher for large n where phase 2 may need more time.")
    p.add_argument("--solver", default="cd195", metavar="NAME",
                   help="SAT solver for RC2 (default: cd195).")
    p.add_argument("--ord-verbose", type=int, default=0, metavar="LEVEL",
                   help="Verbosity level passed to RC2/pathwidth_order during the "
                        "MaxSAT ordering phase (default: 0 = silent). "
                        "Use 10 to see RC2 progress (cost improvements, iteration "
                        "counts) — useful for diagnosing slow or stuck ordering runs.")

    # --- DP options ---
    p.add_argument("--sort-merge", action="store_true",
                   help="Use sort-merge DP backend instead of EH+rsort. "
                        "Run storage (RAM vs file-backed) is chosen automatically "
                        "per step based on available physical RAM.")
    p.add_argument("--cycles", action="store_true",
                   help="Also count Hamiltonian cycles and report alongside paths.")
    p.add_argument("--instrument", action="store_true",
                   help="(sort-merge only) Print per-step phase timing breakdown: "
                        "compute / flush / merge_runs / parallel-merge times per "
                        "worker, imbalance metrics, and dedup ratios.")
    p.add_argument("--mem-reserve-gb", type=float, default=0.0, metavar="GB",
                   help="(sort-merge only) Subtract this many GB from detected "
                        "physical RAM before the sort-merge backend decides whether "
                        "to spill intermediate data to disk (global_ext merge). "
                        "Use this to avoid OOM on large pw=15 graphs: the backend "
                        "will trigger disk spillover earlier, trading time for "
                        "memory safety.  E.g. --mem-reserve-gb 300 on a 768 GB "
                        "machine makes the backend plan for 468 GB available. "
                        "Default: 0 (use all detected RAM).")
    p.add_argument("--on-overflow", choices=["crt", "residue"], default="crt",
                   help="(sort-merge only) If a 64-bit state count wraps: 'crt' "
                        "(default) recovers the exact count by the Chinese "
                        "remainder theorem (replays from the last snapshot before "
                        "the wrap, or full reruns, mod primes < 2^62; certified by "
                        "the Bregman bound); 'residue' stops with the count mod 2^64.")
    p.add_argument("--spill-dir", default=None, metavar="DIR",
                   help="(sort-merge only) Directory for runs that do not fit in "
                        "RAM and for large CRT snapshots (default: $TMPDIR, else "
                        "/tmp).  Should be a real disk, not tmpfs.")
    p.add_argument("--crt-snap-gb", type=float, default=None, metavar="GB",
                   help="(sort-merge only) Most RAM the CRT anchor snapshot may "
                        "use; larger tables are written once to --spill-dir. "
                        "Default: min(RAM/16, 64 GB).")
    p.add_argument("--ram-gb", type=float, default=0.0, metavar="GB",
                   help="(SA ordering only) Physical RAM in GB.  When set, the SA "
                        "cost function adds a quadratic OOM penalty for any "
                        "ordering step whose predicted state count would exceed "
                        "RAM/2/24B entries (the two-buffer introduce threshold). "
                        "Prevents SA from choosing orderings that would OOM during "
                        "the introduce phase.  Recommended: set to ~60%% of total "
                        "RAM to leave headroom for worker buffers and OS, e.g. "
                        "--ram-gb 460 on a 768 GB machine.  Default: 0 (disabled).")
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
    p.add_argument("--machine-yaml", default=None, metavar="PATH",
                   help="Path to machine.yaml produced by tune_params.py "
                        "(e.g. machine_m3pro.yaml). Overrides the default "
                        "search: (1) machine.yaml next to ham_dp_c.py, "
                        "(2) machine.yaml in the current directory. "
                        "The C library recompiles automatically if constants change.")

    # ── Two-pass parse so --config sets defaults before CLI args apply ──────
    # Pass 1: extract --config (ignore everything else, including unknowns).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", metavar="FILE")
    pre_args, _ = pre.parse_known_args()

    if pre_args.config:
        import yaml, sys
        try:
            with open(pre_args.config) as fh:
                cfg = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            p.error(f"--config: file not found: {pre_args.config!r}")
        except yaml.YAMLError as e:
            p.error(f"--config: YAML parse error in {pre_args.config!r}: {e}")

        if not isinstance(cfg, dict):
            p.error(f"--config: top-level value must be a YAML mapping, got {type(cfg).__name__}")

        # Normalise keys: "refine-iters" and "refine_iters" both work.
        # Boolean flags (store_true) written as  verbose: true  in YAML.
        defaults = {}
        for raw_key, val in cfg.items():
            key = str(raw_key).lstrip("-").replace("-", "_")
            defaults[key] = val

        # Validate keys against known destinations so typos surface early.
        known = {a.dest for a in p._actions}
        for k in defaults:
            if k not in known:
                print(f"WARNING: --config: unknown key {k!r} (ignored)", file=sys.stderr)

        p.set_defaults(**defaults)

    # Pass 2: full parse — CLI values override config-file defaults.
    return p.parse_args()


def _ordering_desc(args, use_pw: bool) -> str:
    """Ordering method as printed in the header line."""
    if not use_pw or args.ordering == "bfs":
        return "BFS-heuristic"
    if args.ordering == "kpw":
        return f"kpw-dag({args.kpw_cost})"
    return "MaxSAT-pathwidth"


def _kpw_ordering(label, n_vertices, adj, args, eb, da):
    """Pathwidth and ordering from the kobayashi-pw DAG (see ham_kpw.py)."""
    import re
    cache_label = re.sub(r"[^A-Za-z0-9._-]+", "", label.replace("n=", "G").replace("graph=", ""))
    def get_dag():
        return ham_kpw.load_dag(adj, cache_dir=args.dag_cache or None, kpw_bin=args.kpw_bin,
                                threads=args.kpw_threads, greedy2=args.kpw_greedy2,
                                label=cache_label or "G")
    dag = None
    if args.kpw_engine == "dsl":
        if args.kpw_greedy2 or args.kpw_python:
            print("  WARNING: --kpw-greedy2/--kpw-python apply to --kpw-engine kopt only", flush=True)
        res = ham_kpw.kpw_order(adj, cost=args.kpw_cost, work_model=args.work_model,
                                expand_base=eb, density_alpha=da,
                                max_labels=args.kpw_max_labels, engine="dsl")
    else:
        dag = get_dag()
        res = ham_kpw.kpw_order(adj, cost=args.kpw_cost, work_model=args.work_model,
                                expand_base=eb, density_alpha=da,
                                max_labels=args.kpw_max_labels, dag=dag,
                                python=args.kpw_python, kopt_bin=args.kopt_bin)
    if "fallback" in res.extra:
        print(f"  WARNING: {res.extra['fallback']}", flush=True)
    print(f"  {res.summary()}", flush=True)
    order = res.order

    if args.kpw_candidates > 0:
        dag = dag or get_dag()
        _, cands = ham_kpw.kpw_candidates(
            adj, k=args.kpw_candidates, work_model=args.work_model,
            expand_base=eb, density_alpha=da, max_labels=args.kpw_max_labels, dag=dag,
            python=args.kpw_python, kopt_bin=args.kopt_bin)
        k = args.multi_start_validate or n_vertices // 2
        tags = {tuple(o): t for t, o in cands}
        print(f"  kpw candidates: {len(cands)} distinct; ranking by partial DP "
              f"({k} steps)", flush=True)
        ranked = validate_multistart_orders([(0.0, o) for _, o in cands], adj, n_vertices,
                                            step_limit=k, verbose=True)
        order = list(ranked[0][2])
        print(f"  kpw candidates: chose {tags[tuple(order)]} "
              f"({ranked[0][0]:.0f} ms for {k} steps)", flush=True)
    return res.pw, order


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
    method = args.ordering if use_pw else "bfs"
    eb = getattr(args, 'expand_base', 1.55)
    da = getattr(args, 'density_alpha', 0.25)

    if method == "kpw":
        pw, order = _kpw_ordering(label, n_vertices, adj, args, eb, da)
    elif method == "maxsat":
        try:
            pw, order = get_pathwidth_order(
                G,
                bound=args.bound,
                stratified=args.stratified,
                solver=args.solver,
                minimize_profile=getattr(args, 'minimize_profile', False),
                profile_time_limit=getattr(args, 'profile_time_limit', 120.0),
                verbose=getattr(args, 'ord_verbose', 0),
            )
        except ImportError as e:
            print(f"  WARNING: {e}\n  Falling back to BFS.", flush=True)
            use_pw = False

    if pw is None:
        order = best_bfs_order(adj, n_vertices)

    # --- SA refinement (maxsat; kpw only with --kpw-polish) ---
    if pw is not None and ((method == "maxsat" and not args.no_refine)
                           or (method == "kpw" and args.kpw_polish)):
        sp = getattr(args, 'spike_penalty', 0.0)
        if method == "maxsat" and getattr(args, 'multi_start', False):
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
            _ram_gb = getattr(args, 'ram_gb', 0.0)
            order, best_iter, _ = sa_refine_order(
                                    adj, n_vertices, order, pw,
                                    n_iter=args.refine_iters,
                                    expand_base=eb,
                                    density_alpha=da,
                                    spike_penalty=sp,
                                    ram_gb=_ram_gb,
                                    t_start=getattr(args, 'sa_t_start', 0.0),
                                    t_end=getattr(args, 'sa_t_end', 0.0),
                                    schedule=getattr(args, 'sa_schedule', 'geometric'),
                                    n_restarts=getattr(args, 'sa_restarts', 0),
                                    restart_t_factor=getattr(args, 'sa_restart_factor', 0.5))
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
    _cc = getattr(args, 'cycles', False)
    if args.sort_merge:
        result = count_hamiltonian_paths_sm(
            n_vertices, order, adj,
            verbose=verbose,
            instrument=args.instrument,
            checkpoint_path=ckpt_path,
            checkpoint_secs=args.checkpoint_interval,
            mem_reserve_gb=getattr(args, 'mem_reserve_gb', 0.0),
            count_cycles=_cc,
            on_overflow=getattr(args, 'on_overflow', 'crt'),
            spill_dir=getattr(args, 'spill_dir', None),
            crt_snap_bytes=(None if getattr(args, 'crt_snap_gb', None) is None
                            else int(args.crt_snap_gb * (1 << 30))),
        )
    else:
        result = count_hamiltonian_paths_c(
            n_vertices, order, adj,
            verbose=verbose,
            checkpoint_path=ckpt_path,
            checkpoint_secs=args.checkpoint_interval,
            load_factor=args.load_factor,
            count_cycles=_cc,
        )
    t_dp = time.time() - t_dp

    if _cc and isinstance(result, tuple):
        count, cycles = result
    else:
        count, cycles = result, None

    pw_str = f"pw={pw:2d}" if pw is not None else "pw= ?"
    spike_str = f" [SPIKE +{mx-pw}]" if (pw is not None and mx > pw) else ""
    cyc_str = f"  ham_cycles={cycles}" if cycles is not None else ""
    print(
        f"{label}  {pw_str}  max_fw={mx:2d}  profile={pr:5d}"
        f"  ord={t_ord:.2f}s  dp={t_dp:.3f}s  ham_paths={count}{cyc_str}{spike_str}",
        flush=True,
    )
    return use_pw, count


def main():
    args = parse_args()
    # Apply machine-yaml override before the C library is compiled.
    set_machine_yaml(getattr(args, 'machine_yaml', None))
    use_pw = not args.no_pw and args.ordering != "bfs"

    print("# Compiling C library …", flush=True)
    _get_lib()
    print("# Ready.", flush=True)

    if args.graph:
        # ---------------------------------------------------------------
        # DIMACS mode: single graph file
        # ---------------------------------------------------------------
        path = args.graph
        label = os.path.basename(path)
        print(f"# Graph: {path}  ordering={_ordering_desc(args, use_pw)}",
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
              f"ordering={_ordering_desc(args, use_pw)}",
              flush=True)

        for n in range(start_n, end_n + 1):
            adj = build_graph(n)
            G   = build_gnx(n)
            use_pw, _ = _run_one(f"n={n:3d}", n, G, adj, args, use_pw)


if __name__ == "__main__":
    main()
