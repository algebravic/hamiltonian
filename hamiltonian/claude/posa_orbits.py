"""
posa_orbits_v2.py  —  Posa orbits via C-accelerated path enumeration.

Enumerates Hamiltonian paths in G_n using a C extension for the DFS,
then computes Posa rotation orbits using BFS on the rotation graph.
"""

import ctypes, tempfile, os, hashlib, subprocess, sys
from math import isqrt
from collections import deque

# -------------------------------------------------------------------
# C code: enumerate ALL Hamiltonian paths, write as flat int32 array
# -------------------------------------------------------------------
_C_ENUM = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAXN 64

static int adj[MAXN][MAXN];
static int deg[MAXN];
static int path[MAXN];
static int visited[MAXN];
static int N;

/* Output buffer: each path stored as N int32 values. */
static int32_t *buf = NULL;
static size_t   buf_cap = 0;
static size_t   buf_cnt = 0;   /* number of paths */

static void ensure_buf(void) {
    if (buf_cnt >= buf_cap) {
        buf_cap = buf_cap ? buf_cap * 2 : 65536;
        buf = realloc(buf, buf_cap * N * sizeof(int32_t));
    }
}

static void dfs(int depth) {
    if (depth == N) {
        /* Store canonical path (first < last endpoint). */
        int first = path[0], last = path[N-1];
        ensure_buf();
        int32_t *dst = buf + buf_cnt * N;
        if (first < last) {
            for (int i = 0; i < N; i++) dst[i] = path[i];
        } else {
            for (int i = 0; i < N; i++) dst[i] = path[N-1-i];
        }
        buf_cnt++;
        return;
    }
    int v = path[depth-1];
    for (int k = 0; k < deg[v]; k++) {
        int w = adj[v][k];
        if (!visited[w]) {
            visited[w] = 1;
            path[depth] = w;
            dfs(depth+1);
            visited[w] = 0;
        }
    }
}

/* Called from Python: enumerate all Ham paths, return them as a flat
   int32 array.  *n_paths is set to the count.                        */
int32_t *enumerate_paths(int n, const int *edges, int n_edges, size_t *n_paths_out) {
    N = n;
    memset(deg, 0, sizeof(deg));
    memset(adj, 0, sizeof(adj));
    for (int i = 0; i < n_edges; i++) {
        int u = edges[2*i], v = edges[2*i+1];
        adj[u][deg[u]++] = v;
        adj[v][deg[v]++] = u;
    }
    buf = NULL; buf_cap = 0; buf_cnt = 0;
    for (int s = 1; s <= n; s++) {
        memset(visited, 0, sizeof(visited));
        visited[s] = 1;
        path[0] = s;
        dfs(1);
    }
    /* Remove duplicates: each undirected path appears twice (once from each end).
       Since we always store canonical form, just deduplicate.
       Sort then unique. */
    /* Simple approach: sort lexicographically, then compact. */
    int np = (int)buf_cnt;
    /* qsort with custom comparator */
    int nn = N;
    /* Use insertion-sort-friendly comparison for now; for large np use qsort */
    /* Actually Python will deduplicate, just return all */
    *n_paths_out = buf_cnt;
    return buf;
}

void free_paths(int32_t *p) { free(p); }
"""

_LIB = None
_LIB_N = -1

def _get_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    h = hashlib.md5(_C_ENUM.encode()).hexdigest()[:8]
    so = f'/tmp/posa_enum_{h}.so'
    if not os.path.exists(so):
        src = f'/tmp/posa_enum_{h}.c'
        with open(src, 'w') as f: f.write(_C_ENUM)
        subprocess.check_call(['gcc', '-O3', '-o', so, '-shared', '-fPIC', src])
    lib = ctypes.CDLL(so)
    lib.enumerate_paths.restype  = ctypes.POINTER(ctypes.c_int32)
    lib.enumerate_paths.argtypes = [ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int),
                                     ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_size_t)]
    lib.free_paths.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    _LIB = lib
    return lib


def enumerate_ham_paths_c(n, adj):
    """Return list of canonical path tuples using C DFS."""
    lib = _get_lib()
    # Build edge list
    edges = []
    for u in range(1, n+1):
        for v in adj[u]:
            if u < v:
                edges += [u, v]
    n_edges = len(edges) // 2
    c_edges = (ctypes.c_int * len(edges))(*edges)
    n_paths = ctypes.c_size_t(0)
    ptr = lib.enumerate_paths(n, c_edges, n_edges, ctypes.byref(n_paths))
    np_ = n_paths.value
    # Copy to Python
    raw = [ptr[i] for i in range(np_ * n)]
    lib.free_paths(ptr)
    # Build tuples and deduplicate
    seen = set(); result = []
    for k in range(np_):
        t = tuple(raw[k*n:(k+1)*n])
        if t not in seen:
            seen.add(t); result.append(t)
    return result


# -------------------------------------------------------------------
# Graph
# -------------------------------------------------------------------

def build_adj(n):
    adj = {v: set() for v in range(1, n+1)}
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            s = i+j; r = isqrt(s)
            if r*r == s:
                adj[i].add(j); adj[j].add(i)
    return adj


# -------------------------------------------------------------------
# Posa rotations
# -------------------------------------------------------------------

def posa_rotations(path, adj):
    p = path; n = len(p)
    if n < 4: return
    right = p[-1]
    for i in range(1, n-2):
        if right in adj[p[i]]:
            q = p[:i+1] + tuple(reversed(p[i+1:]))
            yield q if q[0] < q[-1] else tuple(reversed(q))
    rev = tuple(reversed(p)); left = rev[-1]
    for i in range(1, n-2):
        if left in adj[rev[i]]:
            q = rev[:i+1] + tuple(reversed(rev[i+1:]))
            yield q if q[0] < q[-1] else tuple(reversed(q))


# -------------------------------------------------------------------
# Orbit computation
# -------------------------------------------------------------------

def posa_basis_size(n, verbose=False):
    import time
    adj = build_adj(n)

    t0 = time.time()
    paths = enumerate_ham_paths_c(n, adj)
    t_enum = time.time() - t0
    n_paths = len(paths)

    if verbose:
        print(f"  n={n}: {n_paths:,} paths in {t_enum:.2f}s", flush=True)

    if n_paths == 0:
        return 0, 0, []

    path_id = {p: i for i, p in enumerate(paths)}

    # Union-Find
    parent = list(range(n_paths))
    rank   = [0] * n_paths

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py: return
        if rank[px] < rank[py]: px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]: rank[px] += 1

    t0 = time.time()
    for path in paths:
        pid = path_id[path]
        for rot in posa_rotations(path, adj):
            rid = path_id.get(rot)
            if rid is not None:
                union(pid, rid)
    t_rot = time.time() - t0

    if verbose:
        print(f"  rotation graph built in {t_rot:.2f}s", flush=True)

    from collections import Counter
    root_cnt = Counter(find(i) for i in range(n_paths))
    orbit_sizes = sorted(root_cnt.values())
    return len(orbit_sizes), n_paths, orbit_sizes


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("start_n", type=int, nargs="?", default=15)
    p.add_argument("end_n",   type=int, nargs="?", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    start = args.start_n
    end   = args.end_n if args.end_n is not None else start

    print(f"{'n':>4}  {'paths':>10}  {'orbits':>8}  orbit sizes")
    print("-" * 70)
    for n in range(start, end + 1):
        n_orb, n_p, sizes = posa_basis_size(n, verbose=args.verbose)
        print(f"{n:>4}  {n_p:>10,}  {n_orb:>8}  {sizes}")
