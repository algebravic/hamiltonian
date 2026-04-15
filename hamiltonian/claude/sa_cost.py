"""
sa_cost.py — Predict log(states_out / states_in) per DP step.

Supports two backends, loaded from sa_cost_params.json:
  - "linear" : OLS regression (Model B), fast, interpretable
  - "mlp"    : small 2-layer ReLU MLP trained on profile data

Usage (sa_cost.py next to ham_ordering.py):
    from sa_cost import sa_cost_fn
    cost = sa_cost_fn(ordering, adj, n)   # log(total states), lower is better

The JSON is written by profile_analysis.py --save-model <path>.
If no JSON is found a simple fallback proxy is used.
"""

from __future__ import annotations
import json, math, os
from typing import Optional

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_MODEL: Optional[dict] = None
_SEARCH_NAMES = ["sa_cost_params.json"]

def _find_json() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    for name in _SEARCH_NAMES:
        for d in [here, os.getcwd()]:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None

def reload() -> bool:
    """Reload model from disk.  Returns True if a model was found."""
    global _MODEL
    path = _find_json()
    if path is None:
        _MODEL = None
        return False
    with open(path) as f:
        _MODEL = json.load(f)
    return True

def model_info() -> str:
    if _MODEL is None:
        reload()
    if _MODEL is None:
        return "fallback (c^n_back heuristic)"
    kind = _MODEL.get("name", "unknown")
    r2   = _MODEL.get("r2", float("nan"))
    return f"{kind}  R²={r2:.4f}"

# ---------------------------------------------------------------------------
# Feature extraction (same for both backends)
# ---------------------------------------------------------------------------

def _step_features(adj: dict, order: list) -> list[dict]:
    """Return per-step feature dicts for a full ordering."""
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    last_step = {v: max((pos[w] for w in adj.get(v, set())), default=pos[v])
                 for v in range(1, n + 1)}

    features = []
    frontier: set = set()
    for step, v in enumerate(order):
        old_fs = len(frontier)
        frontier.add(v)
        frontier -= {u for u in list(frontier) if last_step[u] <= step}
        new_fs = len(frontier)

        n_back = sum(1 for w in adj.get(v, set()) if w in frontier and pos[w] < step)
        e_bag  = sum(1 for u in frontier for w in adj.get(u, set())
                     if w in frontier and pos[u] < pos[w])

        features.append({
            "fs":       new_fs,
            "delta_fs": new_fs - old_fs,
            "n_back":   n_back,
            "e_bag":    e_bag,
            "fs_nb":    new_fs * n_back,
            "nb_eb":    n_back * e_bag,
            "eb_sq":    e_bag  * e_bag,
            "nb_sq":    n_back * n_back,
            "fs_eb":    new_fs * e_bag,
        })
    return features

# ---------------------------------------------------------------------------
# Linear backend
# ---------------------------------------------------------------------------

def _linear_predict(feat: dict, params: dict) -> float:
    features = params["features"]
    coefs    = params["coefs"]
    total    = coefs[0]  # const / intercept
    for i, fname in enumerate(features[1:], 1):
        total += coefs[i] * feat.get(fname, 0.0)
    return total

# ---------------------------------------------------------------------------
# MLP backend — pure numpy inference
# ---------------------------------------------------------------------------

import struct as _struct

def _relu(x: list[float]) -> list[float]:
    return [max(0.0, v) for v in x]

def _linear_layer(x: list[float], W: list[list[float]], b: list[float]) -> list[float]:
    """Dense layer: out[j] = sum_i x[i]*W[j][i] + b[j]"""
    return [b[j] + sum(x[i] * W[j][i] for i in range(len(x)))
            for j in range(len(b))]

def _mlp_predict(feat: dict, params: dict) -> float:
    feat_names = params["features"]
    x = [feat.get(f, 0.0) for f in feat_names]

    # Standardise inputs
    mu    = params["input_mean"]
    sigma = params["input_std"]
    x = [(x[i] - mu[i]) / (sigma[i] if sigma[i] > 1e-9 else 1.0)
         for i in range(len(x))]

    for layer in params["layers"]:
        x = _linear_layer(x, layer["W"], layer["b"])
        if layer.get("activation") == "relu":
            x = _relu(x)

    return x[0]

# ---------------------------------------------------------------------------
# Cost aggregation (log-sum-exp over steps)
# ---------------------------------------------------------------------------

_WARMUP_STEPS = 3   # skip trivial initial steps

def _aggregate(step_feats: list[dict], predict_fn) -> float:
    """Sum predicted log-ratios into a total log-cost (log-sum-exp)."""
    log_so = 0.0
    log_total = None
    for i, feat in enumerate(step_feats):
        if i < _WARMUP_STEPS or feat["fs"] <= 1:
            continue
        log_ratio = predict_fn(feat)
        log_so += log_ratio
        if log_total is None:
            log_total = log_so
        else:
            m = max(log_total, log_so)
            log_total = m + math.log(math.exp(log_total - m) + math.exp(log_so - m))
    return log_total if log_total is not None else 0.0

# ---------------------------------------------------------------------------
# Fallback: c^n_back proxy (old heuristic)
# ---------------------------------------------------------------------------

def _fallback(step_feats: list[dict]) -> float:
    C = 1.55
    proxy = 1.0; cost = 0.0
    for feat in step_feats:
        expand = C ** feat["n_back"]
        cost += proxy * expand
        proxy = max(proxy * expand, 1.0)
    return math.log(cost + 1.0)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sa_cost_fn(ordering: list, adj: dict, n: int) -> float:
    """
    Predict total DP cost (lower is better) for the given vertex ordering.

    Returns log(total_predicted_states).  Warmup steps (i < 3, fs <= 1)
    are excluded from the prediction.
    """
    global _MODEL
    if _MODEL is None:
        reload()

    feats = _step_features(adj, ordering)

    if _MODEL is None:
        return _fallback(feats)

    kind = _MODEL.get("name", "")
    if "mlp" in kind.lower():
        return _aggregate(feats, lambda f: _mlp_predict(f, _MODEL))
    else:
        return _aggregate(feats, lambda f: _linear_predict(f, _MODEL))


# Auto-load on import
reload()
