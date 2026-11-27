"""
ham_kpw.py — vertex orderings from the kobayashi-pw pathwidth DAG.

Replaces the MaxSAT pathwidth computation and the simulated-annealing
refinement.  ``kpw -c`` (or ``kpw2 -c`` for n > 64) computes the exact
pathwidth pw and a DAG whose source-to-sink paths are pathwidth-optimal
orderings (all orderings that take subsumed vertices eagerly; see
kobayashi-pw/doc/many_orders.md).  A secondary cost is then minimised
exactly over that DAG, in C (``kopt``, via ``kobayashi_pw.orders.optimize_file``)
when the binary is available, else in Python (``kobayashi_pw.orders.optimize``):

  * ``work`` — predicted DP work from the fitted step-local model in
    work_model.py (default);
  * ``v7``   — ham_ordering._dp_cost (the hand-built proxy);
  * ``v2``   — the additive cost sum 2**nb (fw/pw)**2 + 0.001*fw.

Orderings are returned in the frontier-DP direction used by ham_dp_c.

DAG files are cached in ``data/dags`` under a name derived from a hash of the
edge set, so a second run on the same graph skips kpw entirely.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Hashable

from kobayashi_pw.core import Graph
from kobayashi_pw.io import from_edges
from kobayashi_pw.orders import (OrderDAG, find_kopt, k_best, make_model_v7, model_v2,
                                 optimize, optimize_file)

from .ham_ordering import _dp_cost, frontier_width
from .work_model import DEFAULT_PATH as DEFAULT_WORK_MODEL, WorkModel

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DAG_CACHE = os.path.join(_REPO, "data", "dags")
COSTS = ("work", "v7", "v2")


# ---------------------------------------------------------------------------
# Graph and DAG
# ---------------------------------------------------------------------------

def adj_to_graph(adj: dict) -> Graph:
    """kobayashi_pw Graph of an adjacency dict; vertex i has label sorted(adj)[i]."""
    verts = sorted(adj)
    edges = [(u, v) for u in verts for v in adj[u] if u < v]
    return from_edges(edges, vertices=verts)


def _edge_key(g: Graph, greedy2: bool) -> str:
    h = hashlib.sha256()
    h.update(f"n={g.n};greedy2={int(greedy2)};".encode())
    for v in range(g.n):
        h.update(f"{g.labels[v]}:{g.nin[v]:x};".encode())
    return h.hexdigest()[:16]


def find_kpw(n: int, explicit: str | None = None) -> str | None:
    """kpw for n <= 64, kpw2 otherwise: the explicit path, or compiled on first
    use by kobayashi_pw.native (cached under $KPW_CACHE).  None if no C
    compiler is available; the DAG is then built in Python."""
    from kobayashi_pw.native import NativeUnavailable, program
    try:
        return program("kpw" if n <= 64 else "kpw2", n, explicit=explicit)
    except NativeUnavailable:
        if explicit:
            raise
        return None


@dataclass
class DagInfo:
    path: str | None
    cached: bool
    build_s: float
    load_s: float = 0.0           # Python load (only if the DAG was needed in Python)
    nodes: int = 0
    orderings: float = float("nan")
    builder: str = ""


class DagHandle:
    """A DAG file plus, on demand, the DAG loaded in Python (``.dag``).

    Labels: vertex i of the DAG file is ``labels[i]`` (sorted vertices of adj)."""

    def __init__(self, g: Graph, path: str | None, info: DagInfo, dag: OrderDAG | None = None):
        self.g, self.path, self.info, self._dag = g, path, info, dag
        if dag is not None:
            self.pw = dag.pw
        else:
            with open(path, "rb") as f:
                magic = f.read(8)
                n, pw, _, _ = struct.unpack("<4i", f.read(16))
                if magic[:6] != b"KPWDAG" or n != g.n:
                    raise RuntimeError(f"DAG file {path} does not match the graph")
            self.pw = pw

    @property
    def labels(self) -> list:
        return list(self.g.labels)

    @property
    def dag(self) -> OrderDAG:
        if self._dag is None:
            t0 = time.time()
            d = OrderDAG.load(self.path)
            if d.g.n != self.g.n or list(d.g.nin) != list(self.g.nin):
                raise RuntimeError(f"DAG file {self.path} does not match the graph")
            d.g = Graph(self.g.n, list(self.g.nin), list(self.g.labels))
            self._dag = d
            self.info.load_s = time.time() - t0
            self.info.nodes = len(d)
            self.info.orderings = float(d.num_orderings())
        return self._dag


def load_dag(adj: dict, *, cache_dir: str | None = DEFAULT_DAG_CACHE, kpw_bin: str | None = None,
             threads: int = 0, greedy2: bool = False, label: str = "G") -> DagHandle:
    """The pathwidth-optimal DAG of the graph, from the cache or built with kpw.

    Without a kpw binary the DAG is built in Python (much slower) and not cached.
    With caching disabled, the file goes to a temporary directory that lives
    as long as the process."""
    g = adj_to_graph(adj)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{label}_{_edge_key(g, greedy2)}.dag")
    else:
        path = os.path.join(tempfile.mkdtemp(prefix="hamkpw_"), "g.dag")
    t0 = time.time()
    if cache_dir and os.path.isfile(path):
        return DagHandle(g, path, DagInfo(path, True, 0.0, builder="cache"))
    exe = find_kpw(g.n, kpw_bin)
    if exe is None:
        dag = OrderDAG.build(g, greedy2=greedy2)
        info = DagInfo(None, False, time.time() - t0, 0.0, len(dag),
                       float(dag.num_orderings()), "python")
        return DagHandle(g, None, info, dag)
    with tempfile.TemporaryDirectory() as td:
        ef = os.path.join(td, "g.txt")
        with open(ef, "w") as f:                 # vertices renamed 0..n-1, as kpw numbers them
            for v in range(g.n):
                f.write(f"{v}\n")
            for v in range(g.n):
                for u in range(v):
                    if g.nin[v] >> u & 1:
                        f.write(f"{u} {v}\n")
        cmd = [exe, "-q", "-c", "-o", path + ".tmp"] + (["-g"] if greedy2 else [])
        if os.path.basename(exe).startswith("kpw2") and threads != 1:
            cmd += ["-t", str(threads)]
        r = subprocess.run(cmd + [ef], check=True, capture_output=True, text=True)
    os.replace(path + ".tmp", path)
    info = DagInfo(path if cache_dir else None, False, time.time() - t0, builder=os.path.basename(exe))
    for tok in r.stdout.split():
        if tok.startswith("dag="):
            info.nodes = int(tok[4:]) + 1           # + V
    return DagHandle(g, path, info)


# ---------------------------------------------------------------------------
# Costs and exact optima
# ---------------------------------------------------------------------------

def cost_model(cost: str, n: int, *, work_model: str | None = None,
               expand_base: float = 1.55, density_alpha: float = 0.25):
    if cost == "work":
        return WorkModel.load(work_model or DEFAULT_WORK_MODEL).orders_model(n)
    if cost == "v7":
        return make_model_v7(expand_base, density_alpha)
    if cost == "v2":
        return model_v2
    raise ValueError(f"unknown cost {cost!r}; expected one of {COSTS}")


@dataclass
class KpwResult:
    pw: int
    order: list
    cost: str
    value: float
    exact: bool
    max_labels: int
    opt_s: float
    dag: DagInfo
    optimiser: str = "C"
    extra: dict = field(default_factory=dict)
    rounds: list = field(default_factory=list)   # progressive cap: (cap, seconds, value, exact)
    stop: str = ""                               # "exact" or "budget" (progressive cap only)

    def summary(self) -> str:
        d = self.dag
        src = "cached" if d.cached else f"{d.builder} {d.build_s:.1f}s"
        orderings = "" if d.orderings != d.orderings else f" orderings={d.orderings:.2e}"
        load = f" load={d.load_s:.1f}s" if d.load_s else ""
        if d.builder == "dsl":
            src = f"dsl search {d.build_s:.1f}s"
        caps = ""
        if self.rounds:
            caps = " caps=" + ",".join(f"{'none' if c is None else c}:{t:.1f}s/{v:.4g}"
                                       for c, t, v, _ in self.rounds) + f" stop={self.stop}"
        return (f"kpw: pw={self.pw} dag_nodes={d.nodes}{orderings} dag={src}{load} "
                f"opt({self.cost},{self.optimiser})={self.opt_s:.1f}s{caps} "
                f"labels={self.max_labels}"
                f"{'' if self.exact else ' (truncated; not proven optimal)'} "
                f"value={self.value:.4g}")


def check_order(adj: dict, order: list, pw: int) -> None:
    """The order is a permutation of the vertices with max frontier exactly pw."""
    if sorted(order) != sorted(adj):
        raise RuntimeError("kpw ordering is not a permutation of the vertices")
    mx = max(frontier_width(adj, order), default=0)
    if mx != pw:
        raise RuntimeError(f"kpw ordering has max frontier {mx}, expected pw={pw}")


_DSL: dict = {}


def _dsl_problem():
    """vertex_separation with the V2, V7 and log-linear work costs (kobayashi_pw.dsl.hampath)."""
    if "P" not in _DSL:
        from kobayashi_pw.dsl.hampath import hampath_costs
        _DSL["P"] = hampath_costs()
    return _DSL["P"]


def _dsl_order(adj: dict, cost: str, work_model: str | None, expand_base: float,
               density_alpha: float, max_labels: int | None, extra: dict) -> KpwResult:
    """The same optimum computed by the DSL (engine E2, compiled on first use): one process,
    no DAG file; the complete search, pruning and optimisation all happen in generated C."""
    from kobayashi_pw.orders import WORK_FEATURES
    g = adj_to_graph(adj)
    beta = (WorkModel.load(work_model or DEFAULT_WORK_MODEL).beta if cost == "work"
            else [0.0] * len(WORK_FEATURES))
    params = {"base": [expand_base], "alpha": [density_alpha], "beta": beta}
    t0 = time.time()
    s = _dsl_problem().solve(g, output="pareto", secondary=cost, params=params, max_labels=max_labels)
    opt_s = time.time() - t0
    st = s.stats
    info = DagInfo(None, False, st.get("search_seconds", 0.0), nodes=st.get("nodes", 0), builder="dsl")
    order = list(s.order)
    check_order(adj, order, s.width)
    return KpwResult(s.width, order, cost, s.secondary_cost, bool(s.exact), st.get("max_labels", 0),
                     opt_s, info, f"DSL-{s.mode}", extra)


PROGRESSIVE_CAPS = (64, 512, 4096)
DEFAULT_TIME_BUDGET = 60.0


def _progressive(solve_once, budget: float, caps=PROGRESSIVE_CAPS) -> KpwResult:
    """Run solve_once(cap) for cap in caps and then None (no cap), stopping at the
    first result that is not truncated (the exact optimum) or when the next run is
    predicted to exceed the time budget.  The first run always happens.  Every
    result is a pathwidth-optimal ordering; the best value is kept
    (doc/kpw_label_cap_design.md)."""
    rounds, best, used, stop = [], None, 0.0, "budget"
    for cap in list(caps) + [None]:
        if rounds:
            last = rounds[-1][1]
            growth = (last / max(rounds[-2][1], 1e-3)) if len(rounds) >= 2 else 3.0
            if used + last * max(growth, 1.0) > budget:
                break
        t0 = time.time()
        r = solve_once(cap)
        dt = time.time() - t0
        used += dt
        rounds.append((cap, dt, r.value, r.exact))
        if best is None or r.value < best.value or r.exact:
            best = r
        if r.exact:
            stop = "exact"
            break
    best.rounds, best.stop, best.opt_s = rounds, stop, used
    return best


def kpw_order(adj: dict, *, cost: str = "work", work_model: str | None = None,
              expand_base: float = 1.55, density_alpha: float = 0.25,
              max_labels: int | None = None, dag: DagHandle | None = None,
              python: bool = False, kopt_bin: str | None = None, engine: str = "kopt",
              time_budget: float = DEFAULT_TIME_BUDGET, **dag_kw) -> KpwResult:
    """Optimum of ``cost`` over the pathwidth-optimal DAG.

    max_labels: None (default) = progressive cap: caps PROGRESSIVE_CAPS then no cap,
    stopping at the first exact result or when ``time_budget`` seconds (secondary
    optimisation only) would be exceeded; then the best ordering found is returned,
    marked not exact.  0 = no cap, always exact.  L > 0 = a single run with cap L.
"""
    if max_labels is None:
        if engine == "kopt" and dag is None:
            dag = load_dag(adj, **dag_kw)                   # build/load the DAG once
        return _progressive(
            lambda cap: _kpw_order_once(adj, cost=cost, work_model=work_model,
                                        expand_base=expand_base, density_alpha=density_alpha,
                                        max_labels=cap, dag=dag, python=python,
                                        kopt_bin=kopt_bin, engine=engine, **dag_kw),
            time_budget)
    return _kpw_order_once(adj, cost=cost, work_model=work_model, expand_base=expand_base,
                           density_alpha=density_alpha, max_labels=(max_labels or None),
                           dag=dag, python=python, kopt_bin=kopt_bin, engine=engine, **dag_kw)


def _kpw_order_once(adj: dict, *, cost: str = "work", work_model: str | None = None,
                    expand_base: float = 1.55, density_alpha: float = 0.25,
                    max_labels: int | None = None, dag: DagHandle | None = None,
                    python: bool = False, kopt_bin: str | None = None, engine: str = "kopt",
                    **dag_kw) -> KpwResult:
    """One optimisation of ``cost`` over the pathwidth-optimal DAG, with at most
    ``max_labels`` Pareto labels per node (None: no cap, exact).

    engine="kopt" (default): the DAG from kpw/kpw2 (cached), optimised by the C program
    kopt, or in Python if ``python`` is set or kopt cannot be built.
    engine="dsl": the DSL's engine E2 (kobayashi_pw.dsl), which gives the same orderings.
    If cost == "work" and the model file is missing, falls back to v7 and records that in
    ``extra['fallback']``."""
    extra = {}
    if cost == "work" and not os.path.isfile(work_model or DEFAULT_WORK_MODEL):
        extra["fallback"] = f"no work model at {work_model or DEFAULT_WORK_MODEL}; used v7"
        cost = "v7"
    if cost not in COSTS:
        raise ValueError(f"unknown cost {cost!r}; expected one of {COSTS}")
    if engine == "dsl":
        r = _dsl_order(adj, cost, work_model, expand_base, density_alpha, max_labels, extra)
        if cost == "v7":
            ref = _dp_cost(adj, r.order, r.pw, expand_base=expand_base, density_alpha=density_alpha)
            if ref is None or not math.isclose(ref, r.value, rel_tol=1e-9):
                raise RuntimeError(f"V7 mismatch: DSL {r.value} vs _dp_cost {ref}")
        return r
    if engine != "kopt":
        raise ValueError(f"unknown engine {engine!r}; expected 'kopt' or 'dsl'")
    h = dag if dag is not None else load_dag(adj, **dag_kw)
    n = h.g.n
    kopt = None if (python or h.path is None) else find_kopt(kopt_bin)
    t0 = time.time()
    if kopt is not None:
        beta = WorkModel.load(work_model or DEFAULT_WORK_MODEL).beta if cost == "work" else None
        r = optimize_file(h.path, "loglin" if cost == "work" else cost, beta=beta,
                          expand_base=expand_base, density_alpha=density_alpha,
                          max_labels=max_labels, kopt=kopt)
        order = [h.labels[v] for v in r.tau]          # file vertices are 0..n-1
        optimiser = "C"
    else:
        model = cost_model(cost, n, work_model=work_model,
                           expand_base=expand_base, density_alpha=density_alpha)
        r = optimize(h.dag, model, max_labels=max_labels)
        order = list(r.tau)
        optimiser = "python"
    opt_s = time.time() - t0
    check_order(adj, order, h.pw)
    if cost == "v7":
        ref = _dp_cost(adj, order, h.pw, expand_base=expand_base, density_alpha=density_alpha)
        if ref is None or not math.isclose(ref, r.cost, rel_tol=1e-9):
            raise RuntimeError(f"V7 mismatch: optimiser {r.cost} vs _dp_cost {ref}")
    return KpwResult(h.pw, order, cost, r.cost, r.exact, r.max_labels, opt_s, h.info,
                     optimiser, extra)


def kpw_candidates(adj: dict, *, k: int = 5, samples: int = 0, seed: int = 0,
                   costs: tuple[str, ...] = ("work", "v7", "v2"),
                   work_model: str | None = None, expand_base: float = 1.55,
                   density_alpha: float = 0.25, max_labels: int | None = None,
                   dag: DagHandle | None = None, python: bool = False,
                   kopt_bin: str | None = None, time_budget: float = DEFAULT_TIME_BUDGET,
                   **dag_kw) -> tuple[int, list[tuple[str, list]]]:
    """Distinct candidate orderings: the optimum under each cost in ``costs``,
    the k best under V2 (exact; Python), and ``samples`` uniform DAG samples."""
    h = dag if dag is not None else load_dag(adj, **dag_kw)
    out: list[tuple[str, list]] = []
    seen = set()

    def add(tag, order):
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            check_order(adj, list(order), h.pw)
            out.append((tag, list(order)))

    for c in costs:
        if c == "work" and not os.path.isfile(work_model or DEFAULT_WORK_MODEL):
            continue
        r = kpw_order(adj, cost=c, work_model=work_model, expand_base=expand_base,
                      density_alpha=density_alpha, max_labels=max_labels, dag=h,
                      time_budget=time_budget,
                      python=python, kopt_bin=kopt_bin)
        add(f"opt-{c}", r.order)
    if k > 1:
        for i, (_, tau) in enumerate(k_best(h.dag, model_v2, k), 1):
            add(f"v2-rank{i}", tau)
    if samples:
        for i, tau in enumerate(h.dag.sample(random.Random(seed), k=samples), 1):
            add(f"sample{i}", tau)
    return h.pw, out
