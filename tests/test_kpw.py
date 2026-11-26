"""Tests for the kobayashi-pw ordering path (ham_kpw.py, work_model.py)."""

import os
import random

import pytest

from hamiltonian.claude import ham_kpw
from hamiltonian.claude.ham_dp_c import count_hamiltonian_paths_c
from hamiltonian.claude.ham_ordering import _dp_cost, build_graph, frontier_width
from hamiltonian.claude.work_model import DEFAULT_PATH, WorkModel, fit, parse_step_table

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN = {int(a): int(b) for a, b in
         (line.split() for line in open(os.path.join(REPO, "data", "b071983.txt")) if line.strip())}
HAVE_MODEL = os.path.isfile(DEFAULT_PATH)
COSTS = [c for c in ham_kpw.COSTS if c != "work" or HAVE_MODEL]


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return str(tmp_path_factory.mktemp("dags"))


def _dag(n, cache):
    return ham_kpw.load_dag(build_graph(n), cache_dir=cache, label=f"G{n}")


@pytest.mark.parametrize("n", [15, 20, 25, 30, 35, 40, 45])
@pytest.mark.parametrize("cost", COSTS)
def test_order_is_pw_optimal(n, cost, cache):
    adj = build_graph(n)
    r = ham_kpw.kpw_order(adj, cost=cost, dag=_dag(n, cache))
    assert sorted(r.order) == list(range(1, n + 1))
    assert max(frontier_width(adj, r.order)) == r.pw
    assert r.exact


@pytest.mark.parametrize("n", [15, 20, 25, 30, 35, 40])
def test_pw_matches_maxsat(n, cache):
    sep = pytest.importorskip("separation.vertex_separation")
    from hamiltonian.claude.ham_count import build_gnx
    pw_sat, _ = sep.pathwidth_order(build_gnx(n))
    assert _dag(n, cache).pw == pw_sat


@pytest.mark.parametrize("n", [15, 22, 30, 38])
@pytest.mark.parametrize("cost", COSTS)
def test_dp_count(n, cost, cache):
    adj = build_graph(n)
    r = ham_kpw.kpw_order(adj, cost=cost, dag=_dag(n, cache))
    assert count_hamiltonian_paths_c(n, r.order, adj) == KNOWN[n]


@pytest.mark.parametrize("n", [30, 40])
@pytest.mark.parametrize("cost", COSTS)
def test_optimum_beats_samples(n, cost, cache):
    adj = build_graph(n)
    dag = _dag(n, cache)
    r = ham_kpw.kpw_order(adj, cost=cost, dag=dag)
    model = ham_kpw.cost_model(cost, n)
    from kobayashi_pw.orders import evaluate
    for tau in dag.dag.sample(random.Random(1), k=50):
        assert r.value <= evaluate(dag.dag.g, tau, model, dag.pw) * (1 + 1e-9)


def test_v7_equals_dp_cost(cache):
    n = 40
    adj = build_graph(n)
    r = ham_kpw.kpw_order(adj, cost="v7", dag=_dag(n, cache))   # also checked inside
    assert r.value == pytest.approx(_dp_cost(adj, r.order, r.pw), rel=1e-9)


def _all_paths(dag):
    """Every ordering of a small DAG (frontier-DP direction), by enumeration."""
    out = []

    def rec(S, blocks):
        if S == dag.full:
            sigma = list(dag.a0) + [v for b in blocks for v in b]
            out.append([dag.g.labels[v] for v in reversed(sigma)])
            return
        for T, blk in dag.arcs[S]:
            rec(T, blocks + [blk])
    rec(dag.S0, [])
    return out


@pytest.mark.skipif(not HAVE_MODEL, reason="no work_model.json")
def test_work_optimum_is_exact_by_enumeration(cache):
    wm = WorkModel.load()
    rng = random.Random(7)
    graphs = [build_graph(n) for n in (10, 11, 12)]
    while len(graphs) < 40:                       # small random graphs
        n = rng.randint(7, 11)
        adj = {v: set() for v in range(1, n + 1)}
        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                if rng.random() < 0.35:
                    adj[u].add(v); adj[v].add(u)
        graphs.append(adj)
    tried = 0
    for adj in graphs:
        h = ham_kpw.load_dag(adj, cache_dir=None)
        dag = h.dag
        if not 2 <= dag.num_orderings() <= 3000:
            continue
        best = min(wm.predicted_work(dag.g, tau, dag.pw) for tau in _all_paths(dag))
        for python in (False, True):
            r = ham_kpw.kpw_order(adj, cost="work", dag=h, python=python)
            assert r.value == pytest.approx(best, rel=1e-9)
        tried += 1
    assert tried >= 10


def test_candidates(cache):
    n = 35
    adj = build_graph(n)
    pw, cands = ham_kpw.kpw_candidates(adj, k=5, samples=3, dag=_dag(n, cache))
    keys = [tuple(o) for _, o in cands]
    assert len(set(keys)) == len(keys) >= 3
    for _, o in cands:
        assert max(frontier_width(adj, o)) == pw


def test_cache(tmp_path):
    adj = build_graph(33)
    h1 = ham_kpw.load_dag(adj, cache_dir=str(tmp_path), label="G33")
    h2 = ham_kpw.load_dag(adj, cache_dir=str(tmp_path), label="G33")
    if h1.info.builder == "python":
        pytest.skip("no kpw binary: Python-built DAGs are not cached")
    assert not h1.info.cached and h2.info.cached and h1.path == h2.path
    assert ham_kpw.kpw_order(adj, cost="v7", dag=h1).order == \
        ham_kpw.kpw_order(adj, cost="v7", dag=h2).order
    adj2 = build_graph(33)
    u = next(iter(adj2[1]))
    adj2[1].discard(u); adj2[u].discard(1)
    h3 = ham_kpw.load_dag(adj2, cache_dir=str(tmp_path), label="G33")
    assert h3.path != h1.path


def _fake_log(path, n, order):
    """A step table in ham_count's format for G_n with plausible counts."""
    adj = build_graph(n)
    fw = frontier_width(adj, order)
    s = 1
    with open(path, "w") as f:
        f.write("step  vertex  fs  n_back  e_bag  states_in  states_out  step_ms  cumul_ms  bag  [sm]\n")
        for t, v in enumerate(order):
            so = s * (1 + (t * 7 + v) % 5)
            f.write(f"{t:4d} {v:7d} {fw[t]:3d} {t % 3:7d} {1:6d} {s:10d} {so:11d} {0.1:9.1f} {0.1:9.1f}  {{}}  [sm]\n")
            s = so


def test_fit_is_deterministic_and_skips_duplicates(tmp_path, cache):
    logs = []
    for n in (40, 45):
        r = ham_kpw.kpw_order(build_graph(n), cost="v2", dag=_dag(n, cache))
        p = tmp_path / f"profile_{n}.txt"
        _fake_log(p, n, r.order)
        logs.append(str(p))
        assert [row[0] for row in parse_step_table(str(p))] == r.order
    dup = tmp_path / "dup.txt"
    dup.write_text(open(logs[0]).read())
    a = fit(logs + [str(dup)], verbose=False)
    b = fit(logs, verbose=False)
    assert a["beta"] == b["beta"] and len(a["runs"]) == 2
    bad = tmp_path / "bad.txt"                     # frontier column inconsistent with G_n
    _fake_log(bad, 40, list(range(1, 41)))
    rows = open(bad).read().splitlines()
    t = rows[5].split(); t[2] = str(int(t[2]) + 1); rows[5] = "  ".join(t)
    bad.write_text("\n".join(rows) + "\n")
    assert len(fit(logs + [str(bad)], verbose=False)["runs"]) == 2


@pytest.mark.parametrize("n", [40, 50, 57])
@pytest.mark.parametrize("cost", COSTS)
def test_c_and_python_optimisers_agree(n, cost, cache):
    from kobayashi_pw.orders import find_kopt
    if find_kopt() is None:
        pytest.skip("kopt not built")
    adj = build_graph(n)
    h = _dag(n, cache)
    rc = ham_kpw.kpw_order(adj, cost=cost, dag=h)
    rp = ham_kpw.kpw_order(adj, cost=cost, dag=h, python=True)
    assert rc.optimiser == "C" and rp.optimiser == "python"
    assert rc.value == pytest.approx(rp.value, rel=1e-9)
    model = ham_kpw.cost_model(cost, n)
    from kobayashi_pw.orders import evaluate
    assert evaluate(h.dag.g, rc.order, model, h.pw) == pytest.approx(rc.value, rel=1e-9)


@pytest.mark.parametrize("n", [30, 40, 50])
@pytest.mark.parametrize("cost", COSTS)
def test_dsl_engine_equals_kopt(n, cost, cache):
    adj = build_graph(n)
    rk = ham_kpw.kpw_order(adj, cost=cost, dag=_dag(n, cache))
    rd = ham_kpw.kpw_order(adj, cost=cost, engine="dsl")
    assert rd.pw == rk.pw and rd.order == rk.order
    assert rd.value == pytest.approx(rk.value, rel=1e-12)
    if n <= 40:
        assert count_hamiltonian_paths_c(n, rd.order, adj) == KNOWN[n]
