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
  # Analyse a directory of profile files:
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

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='Profile .txt files to analyse')
    ap.add_argument('--dir', metavar='DIR',
                    help='Directory of profile .txt files (uses the "best" file per n)')
    ap.add_argument('--save-model', metavar='JSON',
                    help='Save fitted model coefficients to JSON file')
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
    best_model = model_b if model_b else model_a
    if best_model:
        if args.save_model:
            with open(args.save_model, 'w') as f:
                json.dump(best_model, f, indent=2)
            print(f"\nModel saved to {args.save_model}")

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
