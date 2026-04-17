#!/usr/bin/env python3
"""
profile_analysis.py  --  Regression analysis of DP step profiles
================================================================

PURPOSE
-------
Fit a statistical model of DP state-count growth/collapse as a function
of step-level structural features (fs, n_back, e_bag, delta_fs).  The
fitted model can replace the hand-tuned c^n_back proxy in the SA secondary
objective.

USAGE
-----
  # Analyse a directory of profile files (writes sa_cost_params.json by default):
  python profile_analysis.py --dir /path/to/profiles

  # Analyse specific files:
  python profile_analysis.py profile_68.txt profile_70.txt profile_71.txt ...

  # Also save fitted coefficients for SA integration:
  python profile_analysis.py --dir profiles/ --save-model model.json

PROFILE FILE FORMATS
--------------------
Two formats are supported automatically:

  Old (no e_bag):
    step  vertex  fs  n_back  states_in  states_out  step_ms  cumul_ms

  New (with e_bag, bag label, backend tag):
    step  vertex  fs  n_back  e_bag  states_in  states_out  step_ms  cumul_ms  {bag}  [sm/ext]

  New files also have a summary line:
    n= 70  pw=14  max_fw=14  profile=824  ord=546.33s  dp=1635.171s  ham_paths=...

OUTPUT
------
- Per-n summary table
- Feature correlation analysis
- Regression model: log(so/si) ~ fs + delta_fs + n_back + e_bag + interactions
- Residual diagnostics
- Fitted cost-function coefficients (JSON)
- Comparison: fitted proxy vs current c^n_back heuristic on held-out data
"""

import re
import os
import sys
import json
import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PROFILE PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_profile(text, n_hint=None):
    """
    Parse a single profile log.

    Returns
    -------
    dict with keys:
      n           : graph size (int or None)
      pw          : pathwidth (int or None)
      ham_paths   : Hamiltonian path count (int or None)
      dp_time_s   : total DP time in seconds (float or None)
      ord_time_s  : ordering time in seconds (float or None)
      steps       : list of step dicts (see below)

    Each step dict has:
      step, vertex, fs, n_back, e_bag (None if old format),
      states_in, states_out, step_ms, cumul_ms,
      backend ('sm'|'ext'|'eh'|None)
    """
    n = n_hint
    pw = None
    ham_paths = None
    dp_time_s = None
    ord_time_s = None
    steps = []

    # Summary line: "n= 70  pw=14  max_fw=14  profile=824  ord=546.33s  dp=1635.171s  ham_paths=..."
    m = re.search(r'n=\s*(\d+).*?pw=(\d+).*?ham_paths=(\d+)', text)
    if m:
        n, pw, ham_paths = int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r'dp=([\d.]+)s', text)
    if m:
        dp_time_s = float(m.group(1))
    m = re.search(r'ord=([\d.]+)s', text)
    if m:
        ord_time_s = float(m.group(1))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Skip headers, comments, [inst] lines, SA/warning lines
        if not line:
            continue
        if (line.startswith('#') or line.startswith('[inst]')
                or line.startswith('step') or line.startswith('c ')
                or line.startswith('SA ') or line.startswith('Warning')
                or line.startswith('unstrat') or line.startswith('nohup')):
            continue
        if not line[0].isdigit() and not (line[0] == ' ' and line.strip()[0].isdigit()):
            continue

        parts = line.split()
        if not parts or not parts[0].lstrip('-').isdigit():
            continue

        try:
            # Collect only numeric-looking fields before any {bag} or [backend] token
            vals = []
            for p in parts:
                if p.startswith('{') or p.startswith('['):
                    break
                vals.append(p)

            if len(vals) == 8:
                # Old format: no e_bag
                step   = int(vals[0])
                vertex = int(vals[1])
                fs     = int(vals[2])
                n_back = int(vals[3])
                e_bag  = None
                si     = int(vals[4])
                so     = int(vals[5])
                sms    = float(vals[6])
                cms    = float(vals[7])
                backend = None

            elif len(vals) == 9:
                # New format: has e_bag
                step   = int(vals[0])
                vertex = int(vals[1])
                fs     = int(vals[2])
                n_back = int(vals[3])
                e_bag  = int(vals[4])
                si     = int(vals[5])
                so     = int(vals[6])
                sms    = float(vals[7])
                cms    = float(vals[8])
                bm     = re.search(r'\[(sm|ext|eh)\]', raw_line)
                backend = bm.group(1) if bm else None

            else:
                continue

            steps.append(dict(
                step=step, vertex=vertex, fs=fs, n_back=n_back, e_bag=e_bag,
                states_in=si, states_out=so, step_ms=sms, cumul_ms=cms,
                backend=backend
            ))

        except (ValueError, IndexError):
            continue

    return dict(n=n, pw=pw, ham_paths=ham_paths,
                dp_time_s=dp_time_s, ord_time_s=ord_time_s, steps=steps)


def load_profiles(paths):
    """
    Load and parse a list of profile file paths.
    Returns list of profile dicts (with n inferred from filename if not in file).
    """
    profiles = []
    for path in paths:
        m = re.search(r'_(\d+)', os.path.basename(path))
        n_hint = int(m.group(1)) if m else None
        with open(path) as f:
            text = f.read()
        p = parse_profile(text, n_hint=n_hint)
        if not p['steps']:
            print(f"  WARNING: no steps parsed from {path}", file=sys.stderr)
            continue
        if p['n'] is None:
            print(f"  WARNING: could not determine n for {path}", file=sys.stderr)
            continue
        profiles.append(p)
        status = f"n={p['n']} pw={p['pw']} steps={len(p['steps'])} e_bag={'yes' if p['steps'][0]['e_bag'] is not None else 'no'}"
        print(f"  Loaded {os.path.basename(path)}: {status}")
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD FLAT DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(profiles):
    """
    Convert parsed profiles into a flat DataFrame, one row per DP step.
    Adds derived features: delta_fs, log_ratio, is_terminal, etc.
    Drops steps with states_in <= 1 (warm-up phase, all trivial).
    """
    rows = []
    for p in profiles:
        n   = p['n']
        pw  = p['pw']
        steps = p['steps']

        for i, s in enumerate(steps):
            si = s['states_in']
            so = s['states_out']

            # Skip trivial warm-up (states_in ≤ 1) and terminal (fs=0)
            if si <= 1:
                continue

            # Previous frontier size (for delta_fs)
            prev_fs = steps[i-1]['fs'] if i > 0 else 0

            # Log ratio (the regression target)
            if so == 0 or si == 0:
                log_ratio = None
            else:
                log_ratio = math.log(so / si)

            row = dict(
                n=n,
                pw=pw,
                step=s['step'],
                vertex=s['vertex'],
                fs=s['fs'],
                prev_fs=prev_fs,
                delta_fs=s['fs'] - prev_fs,
                n_back=s['n_back'],
                e_bag=s['e_bag'],          # None for old-format profiles
                states_in=si,
                states_out=so,
                log_states_in=math.log(si),
                log_states_out=math.log(so) if so > 0 else None,
                log_ratio=log_ratio,
                step_ms=s['step_ms'],
                backend=s['backend'],
                has_ebag=(s['e_bag'] is not None),
                is_ext=(s['backend'] == 'ext'),
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    # Derived features
    df['fs_sq'] = df['fs'] ** 2
    df['fs_nb'] = df['fs'] * df['n_back']           # interaction
    df['nb_sq'] = df['n_back'] ** 2
    df['dedup'] = df['states_out'] / df['states_in']

    # For rows with e_bag available
    mask = df['e_bag'].notna()
    df.loc[mask, 'fs_eb']   = df.loc[mask, 'fs'] * df.loc[mask, 'e_bag']
    df.loc[mask, 'nb_eb']   = df.loc[mask, 'n_back'] * df.loc[mask, 'e_bag']
    df.loc[mask, 'eb_sq']   = df.loc[mask, 'e_bag'] ** 2

    # Sequence-aware features (high correlation with log_ratio)
    df['step_frac']   = df['step'] / df['n'].clip(lower=1)       # position in ordering
    df['nb_fs_ratio'] = df['n_back'] / df['fs'].clip(lower=1)    # connectivity density

    # Autoregressive context: previous step's log-ratio (within same profile)
    df = df.sort_values(['n', 'step'])
    df['prev_log_ratio'] = (df.groupby('n')['log_ratio']
                              .shift(1)
                              .fillna(0.0))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_per_n_summary(df, profiles):
    """Print one row per n: peak states, total DP cost, etc."""
    print("\n" + "="*80)
    print("PER-n SUMMARY")
    print("="*80)
    print(f"{'n':>4} {'pw':>3} {'steps':>6} {'peak_states':>14} {'sum_states':>14} "
          f"{'dp_s':>8} {'e_bag':>6}")
    print("─"*65)
    for p in sorted(profiles, key=lambda x: x['n']):
        n = p['n']
        sub = df[df['n'] == n]
        if sub.empty:
            continue
        peak = sub['states_out'].max()
        total = sub['states_out'].sum()
        dp_s = p['dp_time_s']
        has_eb = 'yes' if p['steps'][0]['e_bag'] is not None else 'no'
        dp_str = f"{dp_s:.1f}" if dp_s else "  ?"
        print(f"{n:>4} {p['pw'] or '?':>3} {len(sub):>6} {peak:>14,} {total:>14,} "
              f"{dp_str:>8} {has_eb:>6}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  REGRESSION MODELS
# ─────────────────────────────────────────────────────────────────────────────

def fit_model_no_ebag(df):
    """
    Model A: features available in ALL profiles (old + new format).
    Target: log(states_out / states_in)
    Features: fs, delta_fs, n_back, fs*n_back, n_back^2

    This model can be applied even to old-format profiles (no e_bag).
    """
    sub = df[df['log_ratio'].notna()].copy()

    X = pd.DataFrame({
        'const':    1.0,
        'fs':       sub['fs'],
        'delta_fs': sub['delta_fs'],
        'n_back':   sub['n_back'],
        'fs_nb':    sub['fs_nb'],
        'nb_sq':    sub['nb_sq'],
        'fs_sq':    sub['fs_sq'],
    })
    y = sub['log_ratio']

    return _ols(X, y, name='Model A (no e_bag)')


def fit_model_with_ebag(df):
    """
    Model B: uses e_bag in addition to Model A features.
    Only applicable to new-format profiles.
    """
    sub = df[df['log_ratio'].notna() & df['e_bag'].notna()].copy()
    if len(sub) < 20:
        print("\n  Model B: insufficient e_bag data (need new-format profiles). Skipping.")
        return None

    X = pd.DataFrame({
        'const':    1.0,
        'fs':       sub['fs'],
        'delta_fs': sub['delta_fs'],
        'n_back':   sub['n_back'],
        'e_bag':    sub['e_bag'],
        'fs_nb':    sub['fs_nb'],
        'fs_eb':    sub['fs_eb'],
        'nb_eb':    sub['nb_eb'],
        'nb_sq':    sub['nb_sq'],
        'eb_sq':    sub['eb_sq'],
    })
    y = sub['log_ratio']

    return _ols(X, y, name='Model B (with e_bag)')


def _ols(X, y, name):
    """Fit OLS regression; print results; return result dict."""
    from numpy.linalg import lstsq

    Xm = X.values.astype(float)
    ym = y.values.astype(float)

    coef, residuals, rank, sv = lstsq(Xm, ym, rcond=None)
    y_hat = Xm @ coef
    resid = ym - y_hat
    ss_res = (resid**2).sum()
    ss_tot = ((ym - ym.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    rmse = math.sqrt(ss_res / len(ym))

    # Standard errors via (X'X)^{-1}
    try:
        cov = np.linalg.inv(Xm.T @ Xm) * (ss_res / (len(ym) - len(coef)))
        se = np.sqrt(np.diag(cov))
        t_stat = coef / se
        p_val = 2 * stats.t.sf(np.abs(t_stat), df=len(ym)-len(coef))
    except np.linalg.LinAlgError:
        se = t_stat = p_val = [None]*len(coef)

    print(f"\n{'─'*70}")
    print(f"{name}")
    print(f"  N={len(ym)}  R²={r2:.4f}  RMSE={rmse:.4f}")
    print(f"\n  {'Feature':<12} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
    print(f"  {'─'*50}")
    for feat, c, s, t, pv in zip(X.columns, coef, se, t_stat, p_val):
        sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
        se_s = f"{s:.4f}" if s is not None else "  n/a"
        t_s  = f"{t:.2f}"  if t is not None else "  n/a"
        pv_s = f"{pv:.4f}" if pv is not None else "  n/a"
        print(f"  {feat:<12} {c:>10.4f} {se_s:>10} {t_s:>8} {pv_s:>8} {sig}")

    # Interpretation hint for log-space coefficients
    print(f"\n  Interpretation (multiplicative effect on states_out/states_in):")
    for feat, c in zip(X.columns, coef):
        if feat == 'const':
            print(f"    baseline multiplier per step: ×{math.exp(c):.3f}")
        else:
            print(f"    +1 {feat:<10} → ×{math.exp(c):.4f} (log coef={c:.4f})")

    return dict(
        name=name, features=list(X.columns), coef=coef.tolist(),
        r2=r2, rmse=rmse, n_obs=len(ym)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CURRENT PROXY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_heuristic(df, model_a, c=1.55, alpha=0.25):
    """
    Compare the fitted model's per-step cost prediction against the current
    SA proxy: cost ∝ c^n_back * (1 + alpha * e_bag / fs).

    For each step, compute:
      - actual log(states_out)
      - fitted model prediction
      - current proxy prediction (on log scale, unnormalized)

    Report Pearson r and RMSE for each vs actual.
    """
    sub = df[df['log_ratio'].notna()].copy()

    # Current proxy: log-cost = n_back * log(c) + log(1 + alpha*e_bag/fs)
    # (e_bag term only where available)
    proxy_log = sub['n_back'] * math.log(c)
    mask = sub['e_bag'].notna() & (sub['fs'] > 0)
    proxy_log = proxy_log.copy()
    proxy_log[mask] += np.log(1 + alpha * sub.loc[mask, 'e_bag'] / sub.loc[mask, 'fs'])

    actual = sub['log_ratio'].values

    # Fitted model prediction
    feat_cols = model_a['features']
    Xsub = pd.DataFrame({
        'const':    1.0,
        'fs':       sub['fs'],
        'delta_fs': sub['delta_fs'],
        'n_back':   sub['n_back'],
        'fs_nb':    sub['fs_nb'],
        'nb_sq':    sub['nb_sq'],
        'fs_sq':    sub['fs_sq'],
    })
    fitted = Xsub[feat_cols].values @ np.array(model_a['coef'])

    r_proxy = np.corrcoef(proxy_log.values, actual)[0,1]
    r_model = np.corrcoef(fitted, actual)[0,1]

    rmse_proxy = math.sqrt(((proxy_log.values - actual)**2).mean())
    rmse_model = math.sqrt(((fitted - actual)**2).mean())

    print(f"\n{'─'*70}")
    print("PROXY COMPARISON  (target: log(states_out/states_in) per step)")
    print(f"{'Method':<30} {'Pearson r':>10} {'RMSE':>10}")
    print(f"{'─'*52}")
    print(f"{'Current proxy (c^n_back)':<30} {r_proxy:>10.4f} {rmse_proxy:>10.4f}")
    print(f"{'Fitted Model A':<30} {r_model:>10.4f} {rmse_model:>10.4f}")

    return dict(
        current_proxy=dict(pearson_r=r_proxy, rmse=rmse_proxy),
        model_a=dict(pearson_r=r_model, rmse=rmse_model),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  INTEGRAL vs PEAK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def integral_vs_peak(df, profiles):
    """
    For each n, compare:
      - peak states_out (what pathwidth bounds)
      - sum states_out (what total DP runtime actually scales with)

    Ask: does the ordering that minimises peak also minimise the sum?
    (Can't answer directly without multiple orderings per n, but we can
    look at how tightly peak predicts sum across n-values.)
    """
    print(f"\n{'─'*70}")
    print("INTEGRAL vs PEAK ANALYSIS")
    print(f"\n  {'n':>4} {'peak':>14} {'sum':>14} {'sum/peak':>10} {'dp_s':>8}")
    print(f"  {'─'*55}")
    for p in sorted(profiles, key=lambda x: x['n']):
        n = p['n']
        sub = df[df['n'] == n]
        if sub.empty:
            continue
        peak = sub['states_out'].max()
        total = sub['states_out'].sum()
        dp_s = p['dp_time_s']
        dp_str = f"{dp_s:.1f}" if dp_s else "  ?"
        print(f"  {n:>4} {peak:>14,} {total:>14,} {total/peak:>10.1f} {dp_str:>8}")

    # Check correlation of log(peak) vs log(sum) across n
    ns  = [p['n'] for p in profiles if df[df['n']==p['n']]['states_out'].max() > 1]
    peaks = [df[df['n']==n]['states_out'].max() for n in ns]
    sums  = [df[df['n']==n]['states_out'].sum()  for n in ns]
    if len(ns) >= 3:
        r, pv = stats.pearsonr(np.log(peaks), np.log(sums))
        print(f"\n  Pearson r(log peak, log sum) across n-values: {r:.4f}  (p={pv:.4f})")
        print(f"  → {'High correlation: minimising peak ≈ minimising sum' if r > 0.95 else 'Some divergence: sum and peak can differ'}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SA COST FUNCTION CODE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_sa_cost_function(model):
    """
    Emit Python code for a drop-in SA cost function using the fitted model.

    The SA evaluates cost(ordering) = Σ_t predicted_states_out[t]
    where log(states_out[t]/states_in[t]) ≈ model(features[t]).

    Since states_in[t] isn't known during SA (it's the DP output),
    we use a recursive approximation:
      log(states_out[t]) ≈ log(states_out[t-1]) + model(features[t])
    starting from log(states_out[0]) = 0.

    This is the key insight: the total DP cost prediction becomes a
    running sum of per-step log-ratio predictions, which is fully
    computable from the ordering alone.
    """
    coefs = dict(zip(model['features'], model['coef']))
    c = coefs

    code = f'''
def fitted_sa_cost(ordering, adj, n):
    """
    Predict total DP cost (Σ states_out) from ordering using fitted model.
    Fitted from profile regression: {model["name"]}
    R²={model["r2"]:.4f}  RMSE={model["rmse"]:.4f}  N={model["n_obs"]}

    Uses recursive log-state accumulation:
      log_so[t] = log_so[t-1] + predicted_log_ratio[t]
    starting from log_so = 0 at the trivial first step.

    Parameters
    ----------
    ordering : list of vertex indices (0-indexed), length n
    adj      : adjacency dict  v -> set of neighbours
    n        : number of vertices

    Returns
    -------
    Predicted log(total_states) (lower is better for SA).
    Using log-sum-exp to aggregate avoids overflow.
    """
    import math

    # Compute step features from ordering
    frontier = set()
    log_so = 0.0       # running log(states_out)
    log_total = None   # log-sum-exp accumulator

    for i, v in enumerate(ordering):
        # Compute fs, n_back, e_bag for this step
        new_back_edges = sum(1 for u in adj.get(v, set()) if u in frontier)
        # e_bag: edges within the new bag = back edges (edges to frontier)
        #   Note: for old-format compatibility, e_bag defaults to n_back
        e_bag_val = new_back_edges

        # Update frontier
        old_fs = len(frontier)
        frontier.add(v)
        # Remove vertices whose all neighbours have been introduced
        departed = [u for u in list(frontier) if u != v and
                    all(w in frontier for w in adj.get(u, set()))]
        for u in departed:
            frontier.discard(u)
        new_fs = len(frontier)
        delta_fs = new_fs - old_fs

        fs      = new_fs
        n_back  = new_back_edges
        e_bag   = e_bag_val
        fs_nb   = fs * n_back
        nb_sq   = n_back ** 2
        fs_sq   = fs ** 2

        # Predict log(so/si) for this step
        log_ratio = (
            {c.get("const", 0.0):.6f}
            + {c.get("fs", 0.0):.6f} * fs
            + {c.get("delta_fs", 0.0):.6f} * delta_fs
            + {c.get("n_back", 0.0):.6f} * n_back
            + {c.get("fs_nb", 0.0):.6f} * fs_nb
            + {c.get("nb_sq", 0.0):.6f} * nb_sq
            + {c.get("fs_sq", 0.0):.6f} * fs_sq
        )

        if i < 3 or fs <= 1:
            # Skip warm-up steps (trivial, all states_in=1)
            continue

        log_so = log_so + log_ratio

        # Log-sum-exp accumulation of total predicted cost
        if log_total is None:
            log_total = log_so
        else:
            m = max(log_total, log_so)
            log_total = m + math.log(math.exp(log_total - m) + math.exp(log_so - m))

    return log_total if log_total is not None else 0.0
'''
    return code


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# MLP training (pure numpy — no sklearn/torch required)
# ---------------------------------------------------------------------------

def _relu(x):
    return (x > 0) * x

def _relu_grad(x):
    return (x > 0).astype(float)

def _mlp_forward_col(Xn, Ws, bs, dropout_mask=None, dropout_rate=0.0):
    """Column-format MLP forward. Xn: [N,F] row. Returns (acts_col, pred_Nx1)."""
    acts = [Xn.T]   # [F, N]
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = W @ acts[-1] + b[:, None]
        if i < len(Ws) - 1:
            a = _relu(z)
            if dropout_mask is not None and i == 0 and dropout_rate > 0:
                a = a * dropout_mask.T / (1.0 - dropout_rate)
        else:
            a = z
        acts.append(a)
    return acts, acts[-1].T   # pred: [N,1]

def _mlp_backward_col(resid, acts, Ws, bs, N_total, l2,
                       dropout_mask=None, dropout_rate=0.0):
    """Column-format MLP backward. resid: [N,1]. Returns (dWs, dbs, d_input [N,F])."""
    dz = resid.T / N_total   # [1, N]  — column format
    dWs = [None] * len(Ws)
    dbs_out = [None] * len(bs)
    da = None
    for i in range(len(Ws) - 1, -1, -1):
        dW  = dz @ acts[i].T + l2 / N_total * Ws[i]   # [dim_out, dim_in]
        db  = dz.sum(axis=1)                            # [dim_out]
        dWs[i] = dW
        dbs_out[i] = db
        da = Ws[i].T @ dz                              # [dim_in, N]
        if i > 0:
            gate = (acts[i] > 0).astype(float)
            if dropout_mask is not None and i == 1 and dropout_rate > 0:
                gate = gate * dropout_mask.T / (1.0 - dropout_rate)
            dz = da * gate
        # i=0: da = gradient w.r.t. input (column format [dim_in, N])
    return dWs, dbs_out, da.T  # d_input: [N, dim_in]


def train_mlp(df, hidden=(64, 32), epochs=4000, lr=2e-3, l2=1e-4,
              dropout=0.1, seed=0):
    """
    Train a ReLU MLP on the profile data.

    Features (11):
      9 base: delta_fs, fs, n_back, e_bag, fs_nb, nb_eb, eb_sq, nb_sq, fs_eb
      2 new:  log_si  (= log(states_in), critical: tells model how loaded the DP is)
              prev_log_ratio (= log-ratio of previous step — captures momentum)

    log_si is autoregressive: during SA inference, sa_cost.py substitutes
    the running predicted log(states) so the model "sees" how much state has
    accumulated even without running the actual DP.

    Returns a dict compatible with sa_cost.py.
    """
    import numpy as np

    feat_cols = ['delta_fs', 'fs', 'n_back', 'e_bag',
                 'fs_nb', 'nb_eb', 'eb_sq', 'nb_sq', 'fs_eb',
                 'log_si', 'prev_log_ratio']

    sub = df.copy()
    sub['log_si'] = sub.get('log_states_in',
                             pd.Series(0.0, index=sub.index)).fillna(0.0)
    for f in feat_cols:
        if f not in sub.columns:
            sub[f] = 0.0
    sub = sub.dropna(subset=feat_cols + ['log_ratio'])

    X      = sub[feat_cols].values.astype(float)
    y      = sub['log_ratio'].values.astype(float).reshape(-1, 1)
    n_vals = sub['n'].values
    N, F   = X.shape

    mu    = X.mean(axis=0)
    sigma = X.std(axis=0); sigma[sigma < 1e-9] = 1.0
    Xn    = (X - mu) / sigma

    rng = np.random.default_rng(seed)

    # ── Leave-one-n-out cross-validation ─────────────────────────────────
    unique_ns  = sorted(set(n_vals))
    loo_preds  = np.full(N, np.nan)
    loo_epochs = max(200, epochs // 6)

    for hold_n in unique_ns:
        test_mask  = (n_vals == hold_n)
        train_mask = ~test_mask
        if train_mask.sum() < 20:
            continue
        Xt, yt = Xn[train_mask], y[train_mask]
        Nt = len(yt)
        dims = [F] + list(hidden) + [1]
        _rng = np.random.default_rng(seed + hold_n)
        _Ws = [_rng.standard_normal((dims[i+1], dims[i])) * np.sqrt(2.0/dims[i])
               for i in range(len(dims)-1)]
        _bs = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]
        _mW = [np.zeros_like(W) for W in _Ws]
        _vW = [np.zeros_like(W) for W in _Ws]
        _mb = [np.zeros_like(b) for b in _bs]
        _vb = [np.zeros_like(b) for b in _bs]
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        for ep in range(1, loo_epochs + 1):
            dm = _rng.random((Nt, hidden[0])) > dropout
            acts_, pred_ = _mlp_forward_col(Xt, _Ws, _bs, dm, dropout)
            resid_ = pred_ - yt
            dWs_, dbs_, _ = _mlp_backward_col(resid_, acts_, _Ws, _bs, Nt, l2, dm, dropout)
            for i in range(len(_Ws)):
                _mW[i] = b1*_mW[i] + (1-b1)*dWs_[i]
                _vW[i] = b2*_vW[i] + (1-b2)*dWs_[i]**2
                _mb[i] = b1*_mb[i] + (1-b1)*dbs_[i]
                _vb[i] = b2*_vb[i] + (1-b2)*dbs_[i]**2
                _Ws[i] -= (0.5*lr if ep < loo_epochs//2 else lr) *                           (_mW[i]/(1-b1**ep)) / (np.sqrt(_vW[i]/(1-b2**ep)) + eps_a)
                _bs[i] -= (0.5*lr if ep < loo_epochs//2 else lr) *                           (_mb[i]/(1-b1**ep)) / (np.sqrt(_vb[i]/(1-b2**ep)) + eps_a)
        _, pf = _mlp_forward_col(Xn[test_mask], _Ws, _bs)
        loo_preds[test_mask] = pf.ravel()

    valid = ~np.isnan(loo_preds)
    loo_r2, loo_rmse = float('nan'), float('nan')
    if valid.sum() > 0:
        yv, pv = y[valid].ravel(), loo_preds[valid]
        ss_res = float(np.sum((pv - yv)**2))
        ss_tot = float(np.sum((yv - yv.mean())**2))
        loo_r2   = 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')
        loo_rmse = float(np.sqrt(ss_res/len(yv)))
        print(f"  LOO-CV  R²={loo_r2:.4f}  RMSE={loo_rmse:.4f}  (n_folds={len(unique_ns)})")

    # ── Full training ─────────────────────────────────────────────────────
    dims = [F] + list(hidden) + [1]
    Ws = [rng.standard_normal((dims[i+1], dims[i])) * np.sqrt(2.0/dims[i])
          for i in range(len(dims)-1)]
    bs = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]
    mW = [np.zeros_like(W) for W in Ws]
    vW = [np.zeros_like(W) for W in Ws]
    mb = [np.zeros_like(b) for b in bs]
    vb = [np.zeros_like(b) for b in bs]
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    best_loss, best_params = float('inf'), None

    for epoch in range(1, epochs + 1):
        dm = rng.random((N, hidden[0])) > dropout
        acts, pred = _mlp_forward_col(Xn, Ws, bs, dm, dropout)
        resid = pred - y
        loss  = float(np.mean(resid**2)) + 0.5*l2*sum(float(np.sum(W**2)) for W in Ws)/N
        if loss < best_loss:
            best_loss   = loss
            best_params = ([W.copy() for W in Ws], [b.copy() for b in bs])
        dWs, dbs, _ = _mlp_backward_col(resid, acts, Ws, bs, N, l2, dm, dropout)
        for i in range(len(Ws)):
            mW[i] = b1*mW[i] + (1-b1)*dWs[i]
            vW[i] = b2*vW[i] + (1-b2)*dWs[i]**2
            mb[i] = b1*mb[i] + (1-b1)*dbs[i]
            vb[i] = b2*vb[i] + (1-b2)*dbs[i]**2
            Ws[i] -= lr * (mW[i]/(1-b1**epoch)) / (np.sqrt(vW[i]/(1-b2**epoch)) + eps_a)
            bs[i] -= lr * (mb[i]/(1-b1**epoch)) / (np.sqrt(vb[i]/(1-b2**epoch)) + eps_a)

    Ws_b, bs_b = best_params
    _, pred_train = _mlp_forward_col(Xn, Ws_b, bs_b)
    ss_res = float(np.sum((pred_train - y)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2   = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((pred_train - y)**2)))

    layers = [{"W": W.tolist(), "b": b.ravel().tolist(),
               "activation": "relu" if i < len(Ws_b)-1 else "linear"}
              for i, (W, b) in enumerate(zip(Ws_b, bs_b))]

    return {
        "name":       f"MLP ({len(hidden)} hidden {hidden}, log_si+prev_lr)",
        "features":   feat_cols,
        "input_mean": mu.tolist(),
        "input_std":  sigma.tolist(),
        "layers":     layers,
        "r2":         r2,
        "rmse":       rmse,
        "loo_r2":     loo_r2,
        "n_obs":      N,
    }


def train_conv_mlp(df, kernel_size=5, n_filters=16, hidden=(64, 32),
                   epochs=5000, lr=2e-3, l2=1e-4, dropout=0.1, seed=0):
    """
    Train a 1D ConvMLP on the profile step sequences.

    Architecture
    ------------
    Input features (13):
      9 base: delta_fs, fs, n_back, e_bag, fs_nb, nb_eb, eb_sq, nb_sq, fs_eb
      4 new:  step_frac, nb_fs_ratio, log_si, prev_log_ratio

    Conv1D layer: kernel_size K, n_filters C.  For each step t the conv reads
      a zero-padded window of K consecutive normalised feature vectors and
      produces a C-dim context (patterns like "3 expansions → collapse likely").

    MLP: input = [F features || C conv activations], hidden layers, linear out.

    Backward pass uses consistent column format [dim, N] throughout.
    """
    import numpy as np

    feat_cols = ['delta_fs', 'fs', 'n_back', 'e_bag',
                 'fs_nb', 'nb_eb', 'eb_sq', 'nb_sq', 'fs_eb',
                 'step_frac', 'nb_fs_ratio', 'log_si', 'prev_log_ratio']
    F = len(feat_cols)

    sub = df.copy()
    sub['log_si'] = sub.get('log_states_in',
                             pd.Series(0.0, index=sub.index)).fillna(0.0)
    for col in feat_cols:
        if col not in sub.columns:
            sub[col] = 0.0
    sub = sub.dropna(subset=feat_cols + ['log_ratio', 'n'])

    X_all = sub[feat_cols].values.astype(float)
    mu    = X_all.mean(axis=0)
    sigma = X_all.std(axis=0); sigma[sigma < 1e-9] = 1.0

    sequences = []
    for n_val, grp in sub.groupby('n'):
        grp = grp.sort_values('step')
        Xn = (grp[feat_cols].values.astype(float) - mu) / sigma
        y  = grp['log_ratio'].values.astype(float)
        sequences.append((Xn, y))

    N_total = sum(len(y) for _, y in sequences)

    pad = kernel_size // 2
    def make_windows(Xn):
        N, F_ = Xn.shape
        Xp = np.pad(Xn, ((pad, pad), (0, 0)))
        W  = np.zeros((N, kernel_size * F_))
        for k in range(kernel_size):
            W[:, k*F_:(k+1)*F_] = Xp[k:k+N]
        return W   # [N, K*F]

    rng = np.random.default_rng(seed)
    KF = kernel_size * F
    W_conv = rng.standard_normal((n_filters, KF)) * np.sqrt(2.0 / KF)
    b_conv = np.zeros(n_filters)

    mlp_in_dim = F + n_filters
    dims = [mlp_in_dim] + list(hidden) + [1]
    Ws = [rng.standard_normal((dims[i+1], dims[i])) * np.sqrt(2.0/dims[i])
          for i in range(len(dims)-1)]
    bs = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]

    def _adam_state(p): return np.zeros_like(p), np.zeros_like(p)
    m_Wc, v_Wc = _adam_state(W_conv)
    m_bc, v_bc = _adam_state(b_conv)
    m_Ws = [_adam_state(W) for W in Ws]
    m_bs = [_adam_state(b) for b in bs]
    b1, b2, eps_a = 0.9, 0.999, 1e-8

    best_loss, best_params = float('inf'), None

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        dW_conv_acc = np.zeros_like(W_conv)
        db_conv_acc = np.zeros_like(b_conv)
        dWs_acc = [np.zeros_like(W) for W in Ws]
        dbs_acc = [np.zeros_like(b) for b in bs]

        for Xn, y in sequences:
            N_seq = len(y)
            if N_seq == 0:
                continue

            # ── Forward ──────────────────────────────────────────────────
            Win    = make_windows(Xn)               # [N, K*F]
            conv_z = Win @ W_conv.T + b_conv        # [N, C]
            conv_a = np.maximum(0, conv_z)          # [N, C]
            dmask  = (rng.random(conv_a.shape) > dropout) / (1.0 - dropout)
            conv_ad = conv_a * dmask                # [N, C]

            mlp_in = np.concatenate([Xn, conv_ad], axis=1)  # [N, F+C]

            # MLP forward in column format
            acts = [mlp_in.T]                       # [F+C, N]
            for i, (W, b) in enumerate(zip(Ws, bs)):
                z = W @ acts[-1] + b[:, None]
                acts.append(np.maximum(0, z) if i < len(Ws)-1 else z)

            pred  = acts[-1].T                      # [N, 1]
            resid = pred - y[:, None]               # [N, 1]
            loss  = float(np.mean(resid**2)) / 2
            l2pen = 0.5*l2*(float(np.sum(W_conv**2)) +
                            sum(float(np.sum(W**2)) for W in Ws)) / N_total
            total_loss += loss + l2pen

            # ── Backward (all column format [dim, N]) ─────────────────────
            dz = resid.T / N_total                  # [1, N]

            da = None
            for i in range(len(Ws)-1, -1, -1):
                dW = dz @ acts[i].T + l2/N_total * Ws[i]    # [dim_out, dim_in]
                db = dz.sum(axis=1)                          # [dim_out]
                dWs_acc[i] += dW
                dbs_acc[i] += db
                da = Ws[i].T @ dz                            # [dim_in, N]
                if i > 0:
                    dz = da * (acts[i] > 0)                  # ReLU backward

            # da = gradient w.r.t. mlp_in.T = [F+C, N]
            d_mlp_in = da.T                                  # [N, F+C]
            d_conv_ad = d_mlp_in[:, F:]                      # [N, C]
            d_conv_z  = d_conv_ad * dmask * (conv_z > 0)    # [N, C]

            db_conv_acc += d_conv_z.sum(axis=0)              # [C]
            dW_conv_acc += d_conv_z.T @ Win + l2/N_total * W_conv  # [C, K*F]

        # ── Adam update ───────────────────────────────────────────────────
        def adam(p, g, m, v):
            m[:] = b1*m + (1-b1)*g
            v[:] = b2*v + (1-b2)*g**2
            p -= lr * (m/(1-b1**epoch)) / (np.sqrt(v/(1-b2**epoch)) + eps_a)

        adam(W_conv, dW_conv_acc, m_Wc, v_Wc)
        adam(b_conv, db_conv_acc, m_bc, v_bc)
        for i in range(len(Ws)):
            adam(Ws[i], dWs_acc[i], m_Ws[i][0], m_Ws[i][1])
            adam(bs[i], dbs_acc[i], m_bs[i][0], m_bs[i][1])

        if total_loss < best_loss:
            best_loss   = total_loss
            best_params = (W_conv.copy(), b_conv.copy(),
                           [W.copy() for W in Ws], [b.copy() for b in bs])

        if epoch % 500 == 0:
            print(f"    epoch {epoch:>5}/{epochs}  loss={total_loss:.5f}", flush=True)

    # ── Evaluate ──────────────────────────────────────────────────────────
    W_cb, b_cb, Ws_b, bs_b = best_params
    all_pred, all_y = [], []
    for Xn, y in sequences:
        Win  = make_windows(Xn)
        ca   = np.maximum(0, Win @ W_cb.T + b_cb)       # [N, C]
        inp  = np.concatenate([Xn, ca], axis=1).T       # [F+C, N]
        for i, (W, b) in enumerate(zip(Ws_b, bs_b)):
            inp = W @ inp + b[:, None]
            if i < len(Ws_b)-1:
                inp = np.maximum(0, inp)
        all_pred.append(inp.ravel())    # inp is [1,N] → ravel gives [N]
        all_y.append(y)

    all_pred = np.concatenate(all_pred)
    all_y    = np.concatenate(all_y)
    ss_res = float(np.sum((all_pred - all_y)**2))
    ss_tot = float(np.sum((all_y - all_y.mean())**2))
    r2   = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((all_pred - all_y)**2)))

    layers = [{"W": W.tolist(), "b": b.ravel().tolist(),
               "activation": "relu" if i < len(Ws_b)-1 else "linear"}
              for i, (W, b) in enumerate(zip(Ws_b, bs_b))]

    return {
        "name":             f"ConvMLP (k={kernel_size}, c={n_filters}, {list(hidden)}, log_si)",
        "model_type":       "conv_mlp",
        "features":         feat_cols,
        "input_mean":       mu.tolist(),
        "input_std":        sigma.tolist(),
        "conv_kernel":      W_cb.tolist(),
        "conv_bias":        b_cb.tolist(),
        "conv_kernel_size": kernel_size,
        "conv_n_filters":   n_filters,
        "layers":           layers,
        "r2":               r2,
        "rmse":             rmse,
        "n_obs":            int(N_total),
    }
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='Profile .txt files to analyse')
    ap.add_argument('--dir', metavar='DIR',
                    help='Directory of profile .txt files (uses the "best" file per n)')
    ap.add_argument('--save-model', metavar='JSON', default='sa_cost_params.json',
                    help='Save fitted model coefficients to JSON file '
                         '(default: sa_cost_params.json, ready for sa_cost.py)')
    ap.add_argument('--save-cost-fn', metavar='PY',
                    help='Save generated SA cost function to .py file')
    ap.add_argument('--c', type=float, default=1.55,
                    help='Current proxy base c (default: 1.55)')
    ap.add_argument('--alpha', type=float, default=0.25,
                    help='Current proxy density weight (default: 0.25)')
    ap.add_argument('--best-only', action='store_true',
                    help='When --dir is used, keep only one profile per n '
                         '(prefer complete, prefer newer format)')
    args = ap.parse_args()

    # ── Collect file paths ──────────────────────────────────────────────────
    paths = list(args.files)
    if args.dir:
        d = args.dir
        all_files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                     if f.startswith('profile_') and f.endswith('.txt')]
        if args.best_only:
            # Pick best file per n: prefer complete (reaches fs=0), then newer format
            by_n = defaultdict(list)
            for fp in all_files:
                m = re.search(r'_(\d+)', os.path.basename(fp))
                if m:
                    by_n[int(m.group(1))].append(fp)
            for n in sorted(by_n):
                candidates = by_n[n]
                # Score: complete > has_ebag > shorter name (=simpler filename)
                def score(fp):
                    with open(fp) as f:
                        txt = f.read()
                    complete = 1 if re.search(r'^\s*\d+\s+\d+\s+0\s+', txt, re.M) else 0
                    has_ebag = 1 if re.search(r'^\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+', txt, re.M) else 0
                    return (complete, has_ebag, -len(os.path.basename(fp)))
                paths.append(sorted(candidates, key=score, reverse=True)[0])
        else:
            paths.extend(all_files)

    if not paths:
        print("No profile files specified.  Use positional args or --dir.")
        ap.print_help()
        sys.exit(1)

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"\nLoading {len(paths)} profile file(s)...")
    profiles = load_profiles(paths)
    if not profiles:
        print("No valid profiles loaded.  Exiting.")
        sys.exit(1)

    # Deduplicate by n (keep one per n — prefer more steps, newer format)
    by_n = defaultdict(list)
    for p in profiles:
        by_n[p['n']].append(p)
    deduped = []
    for n in sorted(by_n):
        candidates = by_n[n]
        best = max(candidates,
                   key=lambda p: (p['steps'][0]['e_bag'] is not None, len(p['steps'])))
        deduped.append(best)
    profiles = deduped
    print(f"  → {len(profiles)} distinct n-values: {sorted(p['n'] for p in profiles)}")

    # ── Build dataframe ──────────────────────────────────────────────────────
    df = build_dataframe(profiles)
    print(f"  → {len(df)} step rows after filtering trivial warm-up steps")

    # ── Analyses ─────────────────────────────────────────────────────────────
    print_per_n_summary(df, profiles)
    integral_vs_peak(df, profiles)

    # Feature correlations with log_ratio
    print(f"\n{'─'*70}")
    print("FEATURE CORRELATIONS with log(so/si)")
    sub = df[df['log_ratio'].notna()]
    for feat in ['fs', 'delta_fs', 'n_back', 'fs_nb', 'nb_sq', 'fs_sq']:
        if feat in sub.columns:
            r, p = stats.pearsonr(sub[feat].values, sub['log_ratio'].values)
            print(f"  {feat:<12}: r={r:+.4f}  p={p:.4f}")
    if sub['e_bag'].notna().any():
        for feat in ['e_bag', 'fs_eb', 'nb_eb']:
            if feat in sub.columns:
                sv = sub[sub['e_bag'].notna()]
                r, p = stats.pearsonr(sv[feat].values, sv['log_ratio'].values)
                print(f"  {feat:<12}: r={r:+.4f}  p={p:.4f}  (e_bag rows only)")

    # Regression models
    model_a = fit_model_no_ebag(df)
    model_b = fit_model_with_ebag(df)

    # Proxy comparison
    compare_with_heuristic(df, model_a, c=args.c, alpha=args.alpha)

    # ── Save outputs ─────────────────────────────────────────────────────────
    best_linear = model_b if model_b else model_a

    # Train MLP on same data
    mlp_model = None
    if best_linear:
        print(f"\n{'─'*70}")
        print("Training MLP (64,32) with log_si + prev_log_ratio features...")
        try:
            mlp_model = train_mlp(df, hidden=(64, 32), epochs=4000, lr=2e-3, l2=1e-4)
            loo = mlp_model.get('loo_r2', float('nan'))
            print(f"  MLP train-R²={mlp_model['r2']:.4f}  LOO-R²={loo:.4f}"
                  f"  RMSE={mlp_model['rmse']:.4f}  N={mlp_model['n_obs']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  MLP training failed ({e}), falling back to linear")

    # Train ConvMLP (1D conv + MLP, captures sequential patterns)
    conv_model = None
    print(f"\n{'─'*70}")
    print("Training ConvMLP (k=5, c=16, (64,32)) with log_si + sequence context...")
    try:
        conv_model = train_conv_mlp(df, kernel_size=5, n_filters=16,
                                    hidden=(64, 32), epochs=5000, lr=2e-3, l2=1e-4)
        print(f"  ConvMLP R²={conv_model['r2']:.4f}  RMSE={conv_model['rmse']:.4f}"
              f"  N={conv_model['n_obs']}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ConvMLP training failed ({e})")

    # Choose best model — prefer LOO-CV R² (honest) over training R² (can overfit)
    def _model_score(m):
        if m is None:
            return -float('inf')
        loo = m.get('loo_r2', float('nan'))
        train = m.get('r2', 0.0)
        # If LOO-CV available, use it; otherwise penalise by 0.15 to discourage
        # models that only have training R² (likely overfitting to sequences)
        return loo if not math.isnan(loo) else train - 0.15

    candidates = [m for m in [mlp_model, conv_model, best_linear] if m]
    if candidates:
        best_model = max(candidates, key=_model_score)
        print(f"\n  Model comparison (LOO-R² preferred for selection):")
        for m in [best_linear, mlp_model, conv_model]:
            if m:
                loo  = m.get('loo_r2', float('nan'))
                tr   = m.get('r2', float('nan'))
                loo_s = f"  LOO={loo:.4f}" if not math.isnan(loo) else "  LOO=n/a"
                marker = " <- selected" if m is best_model else ""
                print(f"    {m.get('name','?'):<55} train={tr:.4f}{loo_s}{marker}")
    else:
        best_model = best_linear

    if best_model:
        # Always save — default path is sa_cost_params.json next to this script
        out_path = args.save_model
        with open(out_path, 'w') as f:
            json.dump(best_model, f, indent=2)
        src = best_model.get('name', 'unknown')
        print(f"\nModel saved to {out_path}  [{src}]")
        print(f"  Drop sa_cost_params.json next to sa_cost.py to activate.")

        if args.save_cost_fn:
            code = generate_sa_cost_function(model_a)
            with open(args.save_cost_fn, 'w') as f:
                f.write(code)
            print(f"SA cost function saved to {args.save_cost_fn}")

    # Always print the generated cost function to stdout
    print(f"\n{'='*70}")
    print("GENERATED SA COST FUNCTION (Model A — no e_bag required)")
    print(generate_sa_cost_function(model_a))

    if model_b:
        print(f"\n{'='*70}")
        print("NOTE: Model B (with e_bag) available — better fit when new-format")
        print("profiles are present.  Extend the generated cost function with:")
        eb_coefs = dict(zip(model_b['features'], model_b['coef']))
        for feat in ['e_bag', 'fs_eb', 'nb_eb', 'eb_sq']:
            if feat in eb_coefs:
                print(f"  + {eb_coefs[feat]:.6f} * {feat}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
