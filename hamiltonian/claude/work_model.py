"""
work_model.py — a step-local model of frontier-DP work, fitted to profile logs.

The DP's work for an ordering is

    W = sum_t s_t * 2**nb_t,      s_{t+1} = s_t * exp(g_t),

where s_t is the number of states entering step t, nb_t the number of
back-edges of the vertex added at step t, and g_t = log(states_out/states_in).
This module predicts g_t by least squares from features of the step alone
(the prefix before the step and the vertex added), which are exactly the
quantities ``kobayashi_pw.orders.StepFeatures`` provides on the arcs of the
pathwidth-optimal DAG.  The predicted W then has the (a, e, k) form of
``kobayashi_pw.orders``:

    a = 0,   e = 2**nb,   k = exp(g_hat) / 2**nb

so ``kobayashi_pw.orders.optimize`` minimises it exactly over the DAG (the
cost-to-go is nondecreasing in the carried state count because e >= 0).

The absolute predictions are poorly calibrated: per-step errors compound
along an ordering.  Use the model to rank orderings, not to predict memory.

Fitting
-------
    uv run fit-work-model data/linux/profile_*.txt -o hamiltonian/claude/work_model.json

Each log is a ham_count run on a square-sum graph G_n.  The ordering is read
from the ``vertex`` column of the step table, n from the number of steps,
and the log is used only if the logged frontier sizes agree with G_n under
that ordering.  Logs with identical orderings are used once (state counts
depend only on the graph and the ordering, not on the code version).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from kobayashi_pw.core import Graph
from kobayashi_pw.orders import (WORK_FEATURES, StepFeatures, make_model_loglin, tau_features,
                                 work_features)

from .ham_ordering import build_graph, frontier_width

# The features are defined once, in kobayashi_pw.orders, so that the fit, the
# Python optimiser and the C optimiser (kopt -m loglin) use the same values.
FEATURE_NAMES = WORK_FEATURES

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work_model.json")
MIN_STATES = 1000          # steps with fewer input states are too noisy to fit


def feature_vector(f: StepFeatures, n: int) -> list[float]:
    """Features of one step, in the order of FEATURE_NAMES."""
    return work_features(f, n)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class WorkModel:
    beta: list[float]
    meta: dict

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "WorkModel":
        with open(path) as f:
            d = json.load(f)
        if d.get("features") != FEATURE_NAMES:
            raise ValueError(f"{path}: feature list does not match this code; refit")
        return cls(d["beta"], d)

    def log_ratio(self, f: StepFeatures, n: int) -> float:
        return sum(b * x for b, x in zip(self.beta, feature_vector(f, n)))

    def orders_model(self, n: int) -> Callable[[StepFeatures, int], tuple[float, float, float]]:
        """The (a, e, k) model for kobayashi_pw.orders.optimize on an n-vertex graph."""
        return make_model_loglin(self.beta, n)

    def predicted_work(self, g: Graph, order: Sequence, pw: int) -> float:
        """Predicted W of a full ordering (labels), by the same recursion as optimize."""
        model = self.orders_model(g.n)
        cost, p = 0.0, 1.0
        for f in tau_features(g, order):
            a, e, k = model(f, pw)
            cost += a + p * e
            p = max(p * e * k, 1.0)
        return cost


# ---------------------------------------------------------------------------
# Reading logs
# ---------------------------------------------------------------------------

_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+\s")


def parse_step_table(path: str) -> list[tuple[int, int, int, int, int]]:
    """Rows (vertex, fs, n_back, states_in, states_out) of a ham_count step table."""
    rows, expect = [], 0
    with open(path) as fh:
        for line in fh:
            m = _ROW.match(line)
            if not m:
                continue
            step = int(m[1])
            if step != expect:          # a second table (e.g. a restarted run): keep the first
                break
            rows.append((int(m[2]), int(m[3]), int(m[4]), int(m[6]), int(m[7])))
            expect += 1
    return rows


def square_sum_graph(n: int) -> Graph:
    """G_n as a kobayashi_pw Graph with labels 1..n (vertex i has label i+1)."""
    from kobayashi_pw.io import from_edges
    adj = build_graph(n)
    return from_edges([(u, v) for u in adj for v in adj[u] if u < v], vertices=range(1, n + 1))


def log_observations(path: str) -> tuple[int, list[int], list[list[float]], list[float]] | None:
    """(n, ordering, X, y) from one log, or None if it is not a usable G_n run."""
    rows = parse_step_table(path)
    if not rows:
        return None
    n = len(rows)
    order = [r[0] for r in rows]
    if sorted(order) != list(range(1, n + 1)):
        return None
    if frontier_width(build_graph(n), order) != [r[1] for r in rows]:
        return None                                   # not G_n, or a different convention
    g = square_sum_graph(n)
    X, y = [], []
    for f, (v, fs, nb, si, so) in zip(tau_features(g, order), rows):
        if si >= MIN_STATES and so > 0:
            X.append(feature_vector(f, n))
            y.append(math.log(so / si))
    return n, order, X, y


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit(paths: Iterable[str], verbose: bool = True) -> dict:
    """Least-squares fit on the usable logs; also reports leave-one-n-out R²."""
    import numpy as np
    seen: dict[str, str] = {}
    X, y, grp, runs = [], [], [], []
    for p in sorted(paths):
        obs = log_observations(p)
        if obs is None:
            if verbose:
                print(f"  skip {p}: not a usable G_n step table", file=sys.stderr)
            continue
        n, order, Xi, yi = obs
        key = hashlib.sha1(",".join(map(str, order)).encode()).hexdigest()
        if key in seen:
            if verbose:
                print(f"  skip {p}: same ordering as {seen[key]}", file=sys.stderr)
            continue
        seen[key] = p
        X += Xi; y += yi; grp += [n] * len(yi)
        runs.append({"log": os.path.basename(p), "n": n, "steps_used": len(yi)})
    if not y:
        raise ValueError("no usable logs")
    X, y, grp = np.array(X), np.array(y), np.array(grp)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((X @ beta - y) ** 2)) / ss
    pred = np.zeros_like(y)
    for n in np.unique(grp):
        m = grp == n
        if m.all():
            pred[m] = X[m] @ beta
        else:
            pred[m] = X[m] @ np.linalg.lstsq(X[~m], y[~m], rcond=None)[0]
    r2_cv = 1 - float(np.sum((pred - y) ** 2)) / ss
    return {
        "model": "step-local least squares for log(states_out/states_in)",
        "features": FEATURE_NAMES,
        "beta": [float(b) for b in beta],
        "n_obs": int(len(y)),
        "min_states": MIN_STATES,
        "r2": r2,
        "r2_leave_one_n_out": r2_cv,
        "rmse_leave_one_n_out": float(np.sqrt(np.mean((pred - y) ** 2))),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit the step-local work model to ham_count profile logs.")
    ap.add_argument("logs", nargs="+", help="profile logs (ham_count --profile output)")
    ap.add_argument("-o", "--output", default=DEFAULT_PATH, help=f"JSON output (default {DEFAULT_PATH})")
    a = ap.parse_args(argv)
    d = fit(a.logs)
    with open(a.output, "w") as f:
        json.dump(d, f, indent=1)
    print(f"{len(d['runs'])} runs, {d['n_obs']} steps: R2={d['r2']:.3f}  "
          f"leave-one-n-out R2={d['r2_leave_one_n_out']:.3f}  -> {a.output}")
    for name, b in zip(FEATURE_NAMES, d["beta"]):
        print(f"  {name:22s} {b:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
