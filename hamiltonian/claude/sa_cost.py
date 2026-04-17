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

def _aggregate(step_feats: list[dict], predict_fn,
               ram_gb: float = 0.0) -> float:
    """
    Sum predicted log-ratios into a total log-cost (log-sum-exp).

    If ram_gb > 0, applies a quadratic penalty for any step where the
    predicted state count would exceed the safe introduce threshold
    (ram_gb × 1GB / 2 / 24B).  This prevents SA from choosing orderings
    that would OOM during the two-buffer introduce phase.
    The penalty is large enough to dominate the cost function so SA
    strongly avoids such orderings, but still continuous so the SA
    gradient is smooth.
    """
    ENTRY = 24
    if ram_gb > 0:
        max_safe = (ram_gb * (1 << 30)) / 2 / ENTRY
        log_safe = math.log(max_safe)
        PENALTY = 20.0   # multiplier on quadratic excess
    else:
        log_safe = float('inf')
        PENALTY = 0.0

    log_so = 0.0
    log_total = None
    penalty = 0.0
    for i, feat in enumerate(step_feats):
        if i < _WARMUP_STEPS or feat["fs"] <= 1:
            continue
        log_ratio = predict_fn(feat)
        log_so += log_ratio

        # Penalise if predicted state count after this step would OOM
        if log_so > log_safe:
            excess = log_so - log_safe
            penalty += PENALTY * excess * excess

        if log_total is None:
            log_total = log_so
        else:
            m = max(log_total, log_so)
            log_total = m + math.log(math.exp(log_total - m) + math.exp(log_so - m))

    base = log_total if log_total is not None else 0.0
    return base + penalty

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

def sa_cost_fn(ordering: list, adj: dict, n: int,
               ram_gb: float = 0.0) -> float:
    """
    Predict total DP cost (lower is better) for the given vertex ordering.

    Returns log(total_predicted_states) + OOM penalty.
    Warmup steps (i < 3, fs <= 1) are excluded from the prediction.

    ram_gb: if > 0, adds a quadratic penalty for any step where the
            predicted state count would exceed ram_gb×GB / 2 / 24B
            (the safe threshold for the two-buffer introduce phase).
            Set to the machine's RAM in GB to prevent SA from choosing
            orderings that would OOM.  Example: ram_gb=768.0.
    """
    global _MODEL
    if _MODEL is None:
        reload()

    feats = _step_features(adj, ordering)

    if _MODEL is None:
        return _fallback(feats)

    kind = _MODEL.get("name", "")
    if "mlp" in kind.lower():
        return _aggregate(feats, lambda f: _mlp_predict(f, _MODEL), ram_gb)
    else:
        return _aggregate(feats, lambda f: _linear_predict(f, _MODEL), ram_gb)


# Auto-load on import
reload()

# ---------------------------------------------------------------------------
# Per-step prediction (for model assessment)
# ---------------------------------------------------------------------------

def predict_steps(ordering: list, adj: dict, n: int) -> list[dict]:
    """
    Return per-step predictions for a given ordering.

    Each dict has:
      step, vertex, fs, n_back, e_bag, delta_fs,
      pred_log_ratio   (model prediction of log(so/si))
    """
    global _MODEL
    if _MODEL is None:
        reload()

    feats = _step_features(adj, ordering)

    if _MODEL is None:
        predict_fn = lambda f: math.log(max(1.55 ** f["n_back"], 1.0 + 1e-10))
    elif "mlp" in _MODEL.get("name", "").lower():
        predict_fn = lambda f: _mlp_predict(f, _MODEL)
    else:
        predict_fn = lambda f: _linear_predict(f, _MODEL)

    results = []
    for i, (v, feat) in enumerate(zip(ordering, feats)):
        results.append({
            "step":           i,
            "vertex":         v,
            "fs":             feat["fs"],
            "n_back":         feat["n_back"],
            "e_bag":          feat["e_bag"],
            "delta_fs":       feat["delta_fs"],
            "pred_log_ratio": predict_fn(feat),
        })
    return results


# ---------------------------------------------------------------------------
# Profile comparison
# ---------------------------------------------------------------------------

def _build_adj(n: int) -> dict:
    """Build square-sum adjacency dict for G_n."""
    squares = set(i * i for i in range(2, 2 * n + 2))
    adj: dict = {v: set() for v in range(1, n + 1)}
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if u + v in squares:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _parse_profile(text: str):
    """
    Parse a profile log.  Returns (n, steps_dict).

    steps_dict maps step_index → {vertex, fs, n_back, e_bag, si, so}.
    Handles both old (8-field, no e_bag) and new (9-field) formats.
    """
    import re

    # n from summary line — look for the result summary specifically
    m = re.search(r'n=\s*(\d+)\s+pw=\d+', text)
    if not m:
        m = re.search(r'n=\s*(\d+)', text)
    n = int(m.group(1)) if m else None

    steps: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not line[0].isdigit():
            continue
        # Skip [inst] lines (they start with a digit only after stripping)
        if '[inst]' in raw:
            continue
        parts = line.split()
        # Collect numeric fields before any '{' or '[' token
        nums = []
        for p in parts:
            if p.startswith('{') or p.startswith('['):
                break
            nums.append(p)
        try:
            if len(nums) == 8:                   # old format: no e_bag
                step   = int(nums[0]);  vertex = int(nums[1])
                fs     = int(nums[2]);  n_back = int(nums[3])
                e_bag  = None
                si     = int(nums[4]);  so     = int(nums[5])
            elif len(nums) == 9:                 # new format: has e_bag
                step   = int(nums[0]);  vertex = int(nums[1])
                fs     = int(nums[2]);  n_back = int(nums[3])
                e_bag  = int(nums[4])
                si     = int(nums[5]);  so     = int(nums[6])
            else:
                continue
        except (ValueError, IndexError):
            continue
        steps[step] = {"vertex": vertex, "fs": fs, "n_back": n_back,
                       "e_bag": e_bag, "si": si, "so": so}

    return n, steps


def _features_from_profile_step(step: int, steps: dict) -> dict:
    """
    Construct a feature dict for one step using values from the parsed profile.
    This matches the feature set that _step_features produces, using the
    profile's own fs/n_back/e_bag rather than recomputing from the ordering.
    delta_fs is inferred from the preceding step.
    """
    s     = steps[step]
    fs    = s["fs"]
    nb    = s["n_back"]
    eb    = s["e_bag"] if s["e_bag"] is not None else 0
    prev  = steps.get(step - 1)
    prev_fs = prev["fs"] if prev else 0
    delta = fs - prev_fs
    return {
        "fs":       fs,
        "delta_fs": delta,
        "n_back":   nb,
        "e_bag":    eb,
        "fs_nb":    fs * nb,
        "nb_eb":    nb * eb,
        "eb_sq":    eb * eb,
        "nb_sq":    nb * nb,
        "fs_eb":    fs * eb,
    }


def compare_with_profile(profile_path: str, verbose: bool = True) -> dict:
    """
    Load a profile file and compare MLP predictions against actual log(so/si).

    Features are built directly from the profile's per-step fs/n_back/e_bag,
    matching how the model was trained in profile_analysis.py.

    Returns a dict with aggregate stats (r2, rmse, mae).
    """
    global _MODEL
    if _MODEL is None:
        reload()

    with open(profile_path) as f:
        text = f.read()

    n, steps = _parse_profile(text)
    if n is None or not steps:
        print("ERROR: could not parse profile.")
        return {}

    if _MODEL is None:
        predict_fn = lambda f: math.log(max(1.55 ** f["n_back"], 1.0 + 1e-10))
    elif "mlp" in _MODEL.get("name", "").lower():
        predict_fn = lambda f: _mlp_predict(f, _MODEL)
    else:
        predict_fn = lambda f: _linear_predict(f, _MODEL)

    # Build comparison rows — skip trivial warmup (si <= 1) and terminal (fs=0)
    _WARMUP_SI = 1
    rows = []
    for step in sorted(steps):
        s = steps[step]
        si, so = s["si"], s["so"]
        if si <= _WARMUP_SI:
            continue
        if so == 0:
            continue
        feat   = _features_from_profile_step(step, steps)
        actual = math.log(so / si)
        pred   = predict_fn(feat)
        rows.append({
            "step":   step,
            "vertex": s["vertex"],
            "fs":     s["fs"],
            "n_back": s["n_back"],
            "e_bag":  s["e_bag"],
            "si":     si,
            "so":     so,
            "actual": actual,
            "pred":   pred,
            "error":  pred - actual,
        })

    if not rows:
        print("No comparable steps found (all si <= 1 or so == 0).")
        return {}

    # Aggregate stats
    actuals = [r["actual"] for r in rows]
    errors  = [r["error"]  for r in rows]
    n_rows  = len(rows)
    mean_a  = sum(actuals) / n_rows
    ss_tot  = sum((a - mean_a) ** 2 for a in actuals)
    ss_res  = sum(e ** 2 for e in errors)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse    = math.sqrt(ss_res / n_rows)
    mae     = sum(abs(e) for e in errors) / n_rows

    if verbose:
        import re
        m = re.search(r'n=\s*(\d+).*?dp=([\d.]+)s', text)
        if m:
            print(f"Profile: n={m.group(1)}  dp={float(m.group(2)):.1f}s")
        print(f"Model:   {model_info()}")
        print(f"Steps compared: {n_rows}  (si > 1, so > 0)")
        has_ebag = any(r["e_bag"] is not None for r in rows)
        print()

        # Per-step table
        eb_hdr = "  eb  " if has_ebag else "      "
        hdr = (f"{'step':>4}  {'vtx':>4}  {'fs':>3}  {'nb':>3}{eb_hdr}"
               f"{'si(M)':>8}  {'so(M)':>8}  {'actual':>8}  {'pred':>8}  {'error':>7}")
        print(hdr)
        print("─" * len(hdr))
        for r in rows:
            eb_col = f"  {r['e_bag']:>3}" if has_ebag and r['e_bag'] is not None else "     "
            print(f"{r['step']:>4}  {r['vertex']:>4}  {r['fs']:>3}  {r['n_back']:>3}{eb_col}"
                  f"  {r['si']/1e6:>8.2f}  {r['so']/1e6:>8.2f}"
                  f"  {r['actual']:>8.3f}  {r['pred']:>8.3f}  {r['error']:>+7.3f}")

        print()
        print(f"{'─'*52}")
        print(f"  R²   = {r2:+.4f}   (1.0 = perfect; 0 = predicts mean)")
        print(f"  RMSE = {rmse:.4f}   (in log(so/si) units)")
        print(f"  MAE  = {mae:.4f}")
        print()

        # Worst-predicted steps
        worst = sorted(rows, key=lambda r: abs(r["error"]), reverse=True)[:8]
        print("  Largest prediction errors:")
        print(f"  {'step':>4}  {'nb':>3}  {'fs':>3}  {'actual':>8}  {'pred':>8}  {'error':>7}")
        for r in worst:
            print(f"  {r['step']:>4}  {r['n_back']:>3}  {r['fs']:>3}"
                  f"  {r['actual']:>8.3f}  {r['pred']:>8.3f}  {r['error']:>+7.3f}")
        print()

        # ASCII residual scatter: x=actual, y=error
        print("  Residual plot  (x = actual log(so/si),  y = error = pred − actual):")
        W, H = 60, 12
        xa, xe = min(actuals), max(actuals)
        ye = max(abs(e) for e in errors)
        if ye < 0.01: ye = 0.5
        grid = [[" "] * W for _ in range(H)]
        for r in rows:
            xi = int((r["actual"] - xa) / max(xe - xa, 1e-9) * (W - 1))
            yi = int((r["error"] + ye) / (2 * ye) * (H - 1))
            xi = max(0, min(W - 1, xi))
            yi = max(0, min(H - 1, H - 1 - yi))
            grid[yi][xi] = "·"
        zero_y = H - 1 - int((0 + ye) / (2 * ye) * (H - 1))
        zero_y = max(0, min(H - 1, zero_y))
        for xi in range(W):
            if grid[zero_y][xi] == " ":
                grid[zero_y][xi] = "─"
        for row_i, row in enumerate(grid):
            lbl = ""
            if row_i == 0:      lbl = f"{+ye:+.2f}"
            elif row_i == H//2: lbl = f"{0.0:+.2f}"
            elif row_i == H-1:  lbl = f"{-ye:+.2f}"
            else:               lbl = "     "
            print(f"  {lbl:>6} │{''.join(row)}│")
        print(f"         {xa:.2f}" + " " * (W - 10) + f"{xe:.2f}")
        print(f"         {'actual log(so/si)':^{W}}")

    return {"r2": r2, "rmse": rmse, "mae": mae, "n_steps": n_rows}


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sa_cost.py <profile.txt> [profile2.txt ...]")
        print("  Compares MLP predictions against actual profile step data.")
        sys.exit(0)
    for path in sys.argv[1:]:
        if len(sys.argv) > 2:
            print(f"\n{'='*60}")
            print(f"  {path}")
            print(f"{'='*60}")
        stats = compare_with_profile(path)

