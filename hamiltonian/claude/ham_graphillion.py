"""
ham_graphillion.py
------------------
Improved graphillion interface for counting Hamiltonian paths in G_n.

Key improvement over a naive graphillion call: we compute the best-BFS
vertex ordering and pass the edges to graphillion in a compatible order
(last-endpoint-position sort).  This can reduce ZDD size by orders of
magnitude compared to the natural or default ordering.

Usage
-----
    python ham_graphillion.py 40 56     # count for n = 40..56
"""

from math import isqrt
import time

try:
    import graphillion as gln
    HAS_GRAPHILLION = True
except ImportError:
    HAS_GRAPHILLION = False

from ham_ordering import build_graph, best_order, frontier_stats, graphillion_edge_universe


def count_with_graphillion(n: int, verbose: bool = False) -> int:
    """
    Count undirected Hamiltonian paths in G_n using graphillion.

    The edge universe is sorted according to the best-BFS vertex ordering
    (trying all n start vertices, keeping the one with minimum max
    frontier width, tie-breaking on total profile).

    Returns
    -------
    Exact count as a Python int (graphillion returns a float for large
    values, so we convert via int()).

    Raises
    ------
    ImportError if graphillion is not installed.
    MemoryError (propagated) if the ZDD exceeds available RAM.
    """
    if not HAS_GRAPHILLION:
        raise ImportError("graphillion is not installed. "
                          "Install it with: pip install graphillion")

    adj = build_graph(n)
    order = best_order(adj, n, method="bfs")   # SA can be added for large n

    if verbose:
        mx, pr = frontier_stats(adj, order)
        print(f"  n={n}: best-BFS ordering  max_fw={mx}  profile={pr}", flush=True)

    edges = graphillion_edge_universe(adj, n, order)

    if not edges:
        return 0

    gln.GraphSet.set_universe(edges)
    paths = gln.GraphSet.paths(is_hamilton=True)
    return int(paths.len())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    start_n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    end_n   = int(sys.argv[2]) if len(sys.argv) > 2 else start_n
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if not HAS_GRAPHILLION:
        print("ERROR: graphillion is not installed.")
        sys.exit(1)

    for n in range(start_n, end_n + 1):
        t0 = time.time()
        try:
            count = count_with_graphillion(n, verbose=verbose)
            elapsed = time.time() - t0
            print(f"n={n:3d}  ham_paths={count}  ({elapsed:.2f}s)", flush=True)
        except MemoryError:
            print(f"n={n:3d}  MEMORY ERROR — try ham_frontier_dp.py instead")
            break
