"""Frontier DP vs an independent exact count (Held-Karp), on general graphs.

Regression tests for two bugs found on 2026-11-25:

* back-edges beyond 8 were silently dropped (widxs[8]); a vertex with 9
  earlier neighbours in the frontier gave wrong counts;
* the old 64-bit state key (now ham_dp_c_legacy15.c) held 15 frontier
  slots *including* the vertex being introduced, but the guard allowed 16
  (vertex separation 15), and the 16th slot overlapped the flag bits: wrong
  counts.  The legacy variant now refuses such orderings; the default
  variant ("key16") has 16 slots and no flag bits, and must count them
  exactly (paths and cycles, against Held-Karp and G_n).
"""
import random
import subprocess
import textwrap

import pytest

from hamiltonian.claude import ham_dp_c as m
from hamiltonian.claude.ham_ordering import best_bfs_order, build_graph, frontier_width

HK_C = textwrap.dedent(r"""
    #include <stdio.h>
    #include <stdlib.h>
    int main(void) {
        int n, e; if (scanf("%d %d", &n, &e) != 2) return 1;
        unsigned adj[32] = {0};
        for (int i = 0; i < e; i++) { int a, b; if (scanf("%d %d", &a, &b) != 2) return 1;
            adj[a] |= 1u << b; adj[b] |= 1u << a; }
        size_t N = (size_t)1 << n;
        unsigned __int128 *dp = calloc(N * n, sizeof *dp);
        for (int v = 0; v < n; v++) dp[((size_t)1 << v) * n + v] = 1;
        for (size_t S = 1; S < N; S++) for (int v = 0; v < n; v++) {
            unsigned __int128 c = dp[S * n + v]; if (!c) continue;
            unsigned nb = adj[v] & ~(unsigned)S;
            while (nb) { int w = __builtin_ctz(nb); nb &= nb - 1;
                dp[(S | ((size_t)1 << w)) * n + w] += c; } }
        unsigned __int128 t = 0; for (int v = 0; v < n; v++) t += dp[(N - 1) * n + v];
        printf("%llu\n", (unsigned long long)(t / 2)); return 0; }
""")

RAM = 16 << 30


@pytest.fixture(scope="module")
def hk(tmp_path_factory):
    d = tmp_path_factory.mktemp("hk")
    (d / "hk.c").write_text(HK_C)
    subprocess.run(["cc", "-O2", "-o", str(d / "hk"), str(d / "hk.c")], check=True)
    yml = d / "machine.yaml"
    yml.write_text("SM_NTHREADS: 3\nSM_WORKER_CAP: 8192\nSM_PAR_MERGE_THRESH: 1000\n")
    m.set_machine_yaml(str(yml))

    def count(n, adj):
        edges = [(v - 1, w - 1) for v in adj for w in adj[v] if v < w]
        inp = f"{n} {len(edges)}\n" + "\n".join(f"{a} {b}" for a, b in edges)
        return int(subprocess.run([str(d / "hk")], input=inp, capture_output=True,
                                  text=True, check=True).stdout)
    yield count
    m.set_machine_yaml(None)


def _graphs(seed, count, hk, n_lo=16, n_hi=19):
    """Graphs with Hamiltonian paths whose identity ordering has a hub with
    >= 9 back-edges and an in-step width <= 15."""
    rnd = random.Random(seed)
    out = []
    while len(out) < count:
        n = rnd.randint(n_lo, n_hi)
        adj = {v: set() for v in range(1, n + 1)}
        path = list(range(1, n + 1)); rnd.shuffle(path)   # guarantees H > 0
        for a, b in zip(path, path[1:]):
            adj[a].add(b); adj[b].add(a)
        for v in range(1, n + 1):
            for w in range(v + 1, n + 1):
                if rnd.random() < 0.10:
                    adj[v].add(w); adj[w].add(v)
        hub = n                                   # last vertex: >= 9 back-edges
        for v in rnd.sample(range(1, n), 9):
            adj[hub].add(v); adj[v].add(hub)
        if any(not adj[v] for v in adj):
            continue
        order = list(range(1, n + 1))
        adjl = {v: sorted(a) for v, a in adj.items()}
        if m.in_step_width(order, adjl) <= m.MAX_STATE_SLOTS:
            ref = hk(n, adjl)
            if ref > 0:
                out.append((n, adjl, order, ref))
    return out


@pytest.mark.parametrize("seed", [1, 2])
def test_many_back_edges(hk, seed):
    for n, adj, order, ref in _graphs(seed, 2, hk):
        assert m.count_hamiltonian_paths_sm(n, order, adj, ram_bytes=RAM) == ref
        assert m.count_hamiltonian_paths_c(n, order, adj) == ref


def _perturbed_orders(n, adj, base, widths, per_width, seed=1, tries=6000):
    rnd = random.Random(seed)
    got = {w: [] for w in widths}
    for _ in range(tries):
        o = base[:]
        for _ in range(rnd.randint(1, 14)):
            i, j = rnd.randrange(n), rnd.randrange(n)
            o[i], o[j] = o[j], o[i]
        w = m.in_step_width(o, adj)
        if w in got and len(got[w]) < per_width:
            got[w].append(o)
        if all(len(v) == per_width for v in got.values()):
            break
    return got


def test_gn_widths():
    """G_36: key16 counts in-step width 15 and 16 exactly (paths and cycles)
    and refuses 17; legacy15 counts 15 and refuses 16."""
    n = 36
    adj = build_graph(n)
    base = best_bfs_order(adj, n)
    ref = m.count_hamiltonian_paths_sm(n, base, adj, ram_bytes=RAM, count_cycles=True)
    orders = _perturbed_orders(n, adj, base, (15, 16, 17), 2)
    assert all(orders[w] for w in (15, 16, 17))
    for w in (15, 16):
        for o in orders[w]:
            assert m.count_hamiltonian_paths_sm(n, o, adj, ram_bytes=RAM, count_cycles=True) == ref
            assert m.count_hamiltonian_paths_c(n, o, adj, count_cycles=True) == ref
    for o in orders[17]:
        with pytest.raises(RuntimeError):
            m.count_hamiltonian_paths_sm(n, o, adj, ram_bytes=RAM)
    m.set_c_variant("legacy15")
    try:
        for o in orders[15]:
            assert m.count_hamiltonian_paths_sm(n, o, adj, ram_bytes=RAM, count_cycles=True) == ref
        for o in orders[16]:
            with pytest.raises(RuntimeError):
                m.count_hamiltonian_paths_sm(n, o, adj, ram_bytes=RAM)
            with pytest.raises(RuntimeError):
                m.count_hamiltonian_paths_c(n, o, adj)
    finally:
        m.set_c_variant("key16")


def test_width_16_random_graphs(hk):
    """Random graphs whose identity order needs exactly 16 slots: key16 must
    agree with Held-Karp."""
    rnd = random.Random(7)
    done = 0
    while done < 2:
        n = rnd.randint(17, 18)
        adj = {v: set() for v in range(1, n + 1)}
        path = list(range(1, n + 1)); rnd.shuffle(path)
        for a, b in zip(path, path[1:]):
            adj[a].add(b); adj[b].add(a)
        for v in range(1, n + 1):
            for w in range(v + 1, n + 1):
                if rnd.random() < (0.22 if (v <= 16 < w) else 0.06):
                    adj[v].add(w); adj[w].add(v)
        adjl = {v: sorted(a) for v, a in adj.items()}
        order = list(range(1, n + 1))
        if m.in_step_width(order, adjl) != 16:
            continue
        ref = hk(n, adjl)
        assert m.count_hamiltonian_paths_sm(n, order, adjl, ram_bytes=RAM) == ref
        assert m.count_hamiltonian_paths_c(n, order, adjl) == ref
        done += 1
