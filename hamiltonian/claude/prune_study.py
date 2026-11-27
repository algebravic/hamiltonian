"""prune_study.py — experiments on the completability pruning of the DP.

Design: doc/prune_prediction_design.md.  A Python reference version of the
key16 frontier DP of ham_dp_c.c (same transitions, same canonical labels,
same tests (C) and (D) as sm_prune_classify), used to compute, for a graph
and an ordering, the reachable states R_t (no pruning), the pruned states
P_t, and the live states L_t (states on at least one Hamiltonian path),
plus the decomposition of the endpoint budget.  Small n only.

    python -m hamiltonian.claude.prune_study exact 40 44 --out data/prune_study
    python -m hamiltonian.claude.prune_study orderings 52 --out data/prune_study
    python -m hamiltonian.claude.prune_study analyze --out data/prune_study
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

from .ham_ordering import build_graph

CLOSED = "CLOSED"


# ---------------------------------------------------------------- states
def canon(s: tuple) -> tuple:
    """Relabel positive labels in order of first appearance."""
    m, nxt, out = {}, 1, []
    for x in s:
        if x > 0:
            if x not in m:
                m[x] = nxt
                nxt += 1
            x = m[x]
        out.append(x)
    return tuple(out)


def apply_edge(s, vi, wi, last):
    """ham_dp_c.c apply_edge: None if invalid, CLOSED if the edge closes the
    Hamiltonian path at the last step."""
    if s is CLOSED:
        return None
    sv, sw = s[vi], s[wi]
    if sv == -1 or sw == -1:
        return None
    t = list(s)
    if sv == 0 and sw == 0:
        L = max([x for x in s if x > 0], default=0) + 1
        t[vi] = t[wi] = L
        return canon(t)
    if sv == 0:
        t[wi] = -1
        t[vi] = sw
        return canon(t)
    if sw == 0:
        t[vi] = -1
        t[wi] = sv
        return canon(t)
    if sv == sw:
        return None
    cv, cw = s.count(sv), s.count(sw)
    t[vi] = t[wi] = -1
    t = [sv if x == sw else x for x in t]
    if cv == 1 and cw == 1:
        if not last:
            return None
        return CLOSED if all(x == -1 for x in t) else None
    return canon(t)


def apply_elim(s, elim_desc, last):
    """ham_dp_c.c apply_elim_seq.  Returns (state or None, completes)."""
    if s is CLOSED:
        return None, last
    t = list(s)
    for i in elim_desc:
        su = t[i]
        if su == 0:
            return None, False
        del t[i]
        if su > 0 and su not in t:            # chain's other end left earlier
            if not last:
                return None, False
            return None, all(x == -1 for x in t)
        t = list(canon(t))
    if not t:
        return None, False
    return tuple(t), False


# ---------------------------------------------------------------- pruning
class PruneTables:
    """Per-step tables of sm_prune_setup, for the frontier after step t."""

    def __init__(self, adj, order, pos, step, frontier):
        n = len(order)
        self.active = step < n - 1
        self.fs = len(frontier)
        if not self.active:
            return
        U = order[step + 1:]
        Uset = set(U)
        par = {u: u for u in U}

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for u in U:
            for x in adj[u]:
                if x in Uset:
                    a, b = find(u), find(x)
                    if a != b:
                        par[a] = b
        cid = {}
        for u in U:
            cid.setdefault(find(u), len(cid))
        self.ncomp = len(cid)
        self.allcomp = (1 << self.ncomp) - 1
        sidx = {u: k for k, u in enumerate(frontier)}
        self.fdeg = [0] * self.fs
        self.cmask = [0] * self.fs
        for k, u in enumerate(frontier):
            for x in adj[u]:
                if x in Uset:
                    self.fdeg[k] += 1
                    self.cmask[k] |= 1 << cid[find(x)]
        self.w = []                   # (udeg, amask) for future vertices next to F
        self.e_fixed = 0
        self.all_dead = False
        for w in U:
            ud = sum(1 for x in adj[w] if x in Uset)
            am = 0
            for x in adj[w]:
                if x in sidx:
                    am |= 1 << sidx[x]
            if am:
                self.w.append((ud, am))
            elif ud == 0:
                self.all_dead = True
            elif ud == 1:
                self.e_fixed += 1

    def terms(self, s):
        """(s_dangling, z_untouched_forced, w_forced, w_zero, nonint_mask, labm)."""
        nonint = 0
        labm = defaultdict(int)
        z = 0
        for k, v in enumerate(s):
            if v == -1:
                continue
            nonint |= 1 << k
            if v == 0:
                if self.fdeg[k] == 1:
                    z += 1
            else:
                labm[v] |= 1 << k
        sd = sum(1 for m in labm.values() if m & (m - 1) == 0)
        w1 = w0 = 0
        for ud, am in self.w:
            h = ud + bin(am & nonint).count("1")
            if h == 0:
                w0 += 1
            elif h == 1:
                w1 += 1
        return sd, z, w1 + self.e_fixed, w0, nonint, labm

    def classify(self, s):
        """0 live, 1 dead by (D), 2 dead by (C) — as sm_prune_classify."""
        if not self.active:
            return 0
        if self.all_dead:
            return 1
        sd, z, w1, w0, nonint, labm = self.terms(s)
        if w0 or sd + z + w1 > 2:
            return 1
        if not nonint:
            return 2
        S = nonint & -nonint
        C = 0
        while True:
            C2 = C
            for k in range(self.fs):
                if S >> k & 1:
                    C2 |= self.cmask[k]
            S2 = S
            for k in range(self.fs):
                if (nonint >> k & 1) and not (S >> k & 1) and (self.cmask[k] & C2):
                    S2 |= 1 << k
            for k in range(self.fs):
                if S2 >> k & 1 and s[k] > 0:
                    S2 |= labm[s[k]]
            if S2 == S and C2 == C:
                break
            S, C = S2, C2
        return 0 if (S == nonint and C == self.allcomp) else 2


# ---------------------------------------------------------------- the DP
def run_dp(adj, order, prune=False, record=False, keep_tables=False):
    """Forward pass.  Returns dict with per-step tables (sets of states when
    record), counts, transitions (when record), completions, totals."""
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    last_s = {v: max([pos[w] for w in adj[v]], default=pos[v]) for v in order}
    cur = {(): 1}
    frontier: list = []
    total = 0
    steps = []
    for t, v in enumerate(order):
        last = t == n - 1
        frontier = frontier + [v]
        vi = len(frontier) - 1
        widx = [k for k, w in enumerate(frontier) if w in adj[v] and pos[w] < t]
        elim_asc = [k for k, u in enumerate(frontier) if last_s[u] <= t]
        elim_desc = elim_asc[::-1]
        post = [u for k, u in enumerate(frontier) if k not in set(elim_asc)]
        pt = PruneTables(adj, order, pos, t, post) if prune or record else None
        nxt: dict = defaultdict(int)
        trans = {} if record else None
        completes = set() if record else None
        generated = 0
        dropped = 0
        for s0, c in cur.items():
            s = s0 + (0,)
            succ = set() if record else None
            for S in range(1 << len(widx)):
                k = s
                ok = True
                for j, wi in enumerate(widx):
                    if S >> j & 1:
                        k = apply_edge(k, vi, wi, last)
                        if k is None:
                            ok = False
                            break
                if not ok:
                    continue
                if k is CLOSED or elim_desc:
                    k, done = apply_elim(k, elim_desc, last)
                    if done:
                        total += c
                        if record:
                            completes.add(s0)
                    if k is None:
                        continue
                generated += 1
                if prune and pt.classify(k):
                    dropped += 1
                    continue
                nxt[k] += c
                if record:
                    succ.add(k)
            if record:
                trans[s0] = succ
        steps.append(dict(step=t, v=v, fs=len(post), n_back=len(widx),
                          states=len(nxt), generated=generated, dropped=dropped,
                          table=set(nxt) if (record or keep_tables) else None, trans=trans,
                          completes=completes, pt=pt))
        cur = dict(nxt)
        frontier = post
    return dict(total=total, steps=steps)


def live_sets(res):
    """Backward pass: L_t = states of R_t on some Hamiltonian path."""
    steps = res["steps"]
    n = len(steps)
    live = [set() for _ in range(n)]
    # states of table t-1 are keys of trans at step t
    for t in range(n - 1, 0, -1):
        st = steps[t]
        L = set(st["completes"])
        for s0, succ in st["trans"].items():
            if succ & live[t]:
                L.add(s0)
        live[t - 1] = L
    return live


# ---------------------------------------------------------------- drivers
def exact(n, order, adj):
    t0 = time.time()
    R = run_dp(adj, order, prune=False, record=True)
    t1 = time.time()
    P = run_dp(adj, order, prune=True, keep_tables=True)
    L = live_sets(R)
    rows = []
    for t, st in enumerate(R["steps"]):
        pt = st["pt"]
        Rt = st["table"]
        # composition of R_t and the independence predictor of the dead fraction
        joint = Counter()
        dead_true = 0
        if pt is not None and pt.active and Rt:
            for s in Rt:
                sd, z, w1, w0, _, _ = pt.terms(s)
                joint[(sd, z, w1, w0 > 0)] += 1
                dead_true += pt.classify(s) != 0
            N = len(Rt)
            ms = Counter(); mz = Counter(); mw = Counter(); m0 = 0
            for (a, b, c, d), k in joint.items():
                ms[a] += k; mz[b] += k; mw[c] += k; m0 += k * d
            p0 = m0 / N
            p_ok = 0.0
            for a, pa in ms.items():
                for b, pb in mz.items():
                    for c, pc in mw.items():
                        if a + b + c <= 2:
                            p_ok += pa * pb * pc / N ** 3
            pred_dead = 1 - p_ok * (1 - p0)
            dD = sum(k for (a, b, c, d), k in joint.items() if d or a + b + c > 2)
            share = dict(
                s=sum(k for (a, b, c, d), k in joint.items() if a > 2) / max(dD, 1),
                z=sum(k for (a, b, c, d), k in joint.items() if b > 2) / max(dD, 1),
                w=sum(k for (a, b, c, d), k in joint.items() if c > 2) / max(dD, 1))
            mean = dict(s=sum(a * k for a, k in ms.items()) / N,
                        z=sum(b * k for b, k in mz.items()) / N,
                        w=sum(c * k for c, k in mw.items()) / N)
        else:
            pred_dead, share, mean = None, None, None
        rows.append(dict(step=t, fs=st["fs"], n_back=st["n_back"],
                         R=len(Rt), P=P["steps"][t]["states"], L=len(L[t]),
                         gen_R=st["generated"], gen_P=P["steps"][t]["generated"],
                         dead_frac=(dead_true / len(Rt)) if Rt and pred_dead is not None else None,
                         pred_dead_indep=pred_dead, single_term_share=share, mean_terms=mean))
        assert L[t] <= P["steps"][t]["table"] <= Rt, t
    assert R["total"] == P["total"], (R["total"], P["total"])
    return dict(n=n, total=R["total"], order=order, rows=rows,
                time_R=t1 - t0, time_all=time.time() - t0)


def summarize(res):
    rows = res["rows"]
    SR = sum(r["R"] for r in rows); SP = sum(r["P"] for r in rows); SL = sum(r["L"] for r in rows)
    GR = sum(r["gen_R"] for r in rows); GP = sum(r["gen_P"] for r in rows)
    return dict(n=res["n"], total=res["total"], sum_R=SR, sum_P=SP, sum_L=SL,
                rho=SR / SP, gap=SP / SL, ceiling=SR / SL, gen_ratio=GR / GP,
                peak_R=max(r["R"] for r in rows), peak_P=max(r["P"] for r in rows),
                peak_L=max(r["L"] for r in rows))


# ---------------------------------------------------------------- features
def ordering_features(adj, order):
    """Static features of (G, order) — doc/prune_prediction_design.md §2."""
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    last_s = {v: max([pos[w] for w in adj[v]], default=pos[v]) for v in order}
    F: list = []
    sumf = maxf = O = W = soj = soj1 = 0
    for t, v in enumerate(order):
        F = [u for u in F + [v] if last_s[u] > t]
        U = set(order[t + 1:])
        f = len(F)
        sumf += f
        maxf = max(maxf, f)
        soj += f
        for u in F:
            du = sum(1 for x in adj[u] if x in U)
            if du == 1:
                O += 1
                soj1 += 1
        Fs = set(F)
        for w in U:
            if any(x in Fs for x in adj[w]):
                if sum(1 for x in adj[w] if x in U or x in Fs) <= 2:
                    W += 1
    return dict(sum_f=sumf, max_f=maxf, forced_opp=O, lowdeg_exposure=W,
                sojourn_one_left=soj1)


# ---------------------------------------------------------------- experiment 2
_C_CHILD = """
import sys, json
from hamiltonian.claude import ham_dp_c as m
from hamiltonian.claude.ham_ordering import build_graph
n, mode, order = int(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3])
print(m.count_hamiltonian_paths_sm(n, order, build_graph(n), verbose=True, prune=mode))
"""


def run_c(n, order, mode, timeout):
    """Run the C DP on one ordering; per-step states_out and DP time (ms)."""
    import re
    import subprocess
    try:
        p = subprocess.run([sys.executable, "-c", _C_CHILD, str(n), mode, json.dumps(order)],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(timeout=True)
    steps = [l.split() for l in p.stderr.splitlines()
             if re.match(r"^\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+", l)]
    states = [int(x[6]) for x in steps]
    return dict(timeout=False, total=int(p.stdout.strip().splitlines()[-1]),
                states=states, sum_states=sum(states), peak=max(states),
                dp_ms=float(steps[-1][8]))


def orderings_experiment(n, out, k=3, samples=8, seed=1, timeout=150.0,
                         work_model="work_model.json"):
    """Experiment 2 for one n.  Resumable: one JSON file per (ordering, mode)."""
    from . import ham_kpw
    adj = {v: set(w) for v, w in build_graph(n).items()}
    cpath = os.path.join(out, f"cands_{n}.json")
    if os.path.exists(cpath):
        cands = json.load(open(cpath))
    else:
        pw, c = ham_kpw.kpw_candidates(adj, k=k, samples=samples, seed=seed,
                                       cache_dir=None, work_model=work_model)
        cands = dict(pw=pw, orders=[dict(tag=t, order=list(o),
                                         features=ordering_features(adj, list(o)))
                                    for t, o in c])
        json.dump(cands, open(cpath, "w"))
    for c in cands["orders"]:
        for mode in ("drop", "none"):
            path = os.path.join(out, f"ord_{n}_{c['tag']}_{mode}.json")
            if os.path.exists(path):
                continue
            r = run_c(n, c["order"], mode, timeout)
            r.update(n=n, tag=c["tag"], mode=mode)
            json.dump(r, open(path, "w"))
            print(n, c["tag"], mode, r.get("sum_states"), r.get("dp_ms"),
                  "TIMEOUT" if r["timeout"] else "", flush=True)


def analyze(out):
    """Tables for the write-up: experiment 1 and experiment 2."""
    import glob
    print("## Experiment 1 (work-optimal ordering)")
    print("| n | ρ | gap | ceiling | dead % of R | independence prediction % | ρ predicted |"
          " Σ R·(1-d) / Σ P | share of dead with s>2 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(out, "exact_*.json")),
                    key=lambda x: int(x.rsplit("_", 1)[1][:-5])):
        d = json.load(open(f)); S = d["summary"]
        rows = [x for x in d["rows"] if x["dead_frac"] is not None and x["R"] > 0]
        R = sum(x["R"] for x in rows)
        dt = sum(x["R"] * x["dead_frac"] for x in rows) / R
        dp = sum(x["R"] * x["pred_dead_indep"] for x in rows) / R
        pas = sum(x["R"] * (1 - x["dead_frac"]) for x in rows)
        P = sum(x["P"] for x in rows)
        sh = sum(x["R"] * x["dead_frac"] * x["single_term_share"]["s"] for x in rows) / (dt * R)
        print(f"| {S['n']} | {S['rho']:.2f} | {S['gap']:.2f} | {S['ceiling']:.2f} | {100*dt:.1f} |"
              f" {100*dp:.1f} | {1/(1-dp):.2f} | {pas/P:.3f} | {sh:.2f} |")
    print()
    print("## Experiment 2 (orderings)")
    for cp in sorted(glob.glob(os.path.join(out, "cands_*.json"))):
        n = int(cp.rsplit("_", 1)[1][:-5])
        cands = json.load(open(cp))
        print(f"### n = {n} (pw = {cands['pw']})")
        print("| ordering | Σ states none | Σ states drop | ρ | peak none | peak drop |"
              " time none (s) | time drop (s) | Σf | forced opp. | low-deg exp. |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cands["orders"]:
            r = {}
            for mode in ("none", "drop"):
                p = os.path.join(out, f"ord_{n}_{c['tag']}_{mode}.json")
                r[mode] = json.load(open(p)) if os.path.exists(p) else None
            def g(m, key, scale=1.0, fmt="{:.0f}"):
                x = r[m]
                if x is None:
                    return "—"
                if x.get("timeout"):
                    return "timeout"
                return fmt.format(x[key] / scale)
            rho = "—"
            if r["none"] and r["drop"] and not r["none"].get("timeout") and not r["drop"].get("timeout"):
                rho = f"{r['none']['sum_states']/r['drop']['sum_states']:.2f}"
            f = c["features"]
            print(f"| {c['tag']} | {g('none','sum_states')} | {g('drop','sum_states')} | {rho} |"
                  f" {g('none','peak')} | {g('drop','peak')} | {g('none','dp_ms',1000,'{:.1f}')} |"
                  f" {g('drop','dp_ms',1000,'{:.1f}')} | {f['sum_f']} | {f['forced_opp']} |"
                  f" {f['lowdeg_exposure']} |")
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["exact", "orderings", "analyze"])
    ap.add_argument("start", type=int, nargs="?")
    ap.add_argument("end", type=int, nargs="?")
    ap.add_argument("--out", default="data/prune_study")
    ap.add_argument("--work-model", default="work_model.json")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=150.0)
    a = ap.parse_args(argv)
    from . import ham_kpw
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "analyze":
        analyze(a.out)
        return
    if a.mode == "orderings":
        for n in range(a.start, (a.end or a.start) + 1):
            orderings_experiment(n, a.out, k=a.k, samples=a.samples, timeout=a.timeout,
                                 work_model=a.work_model)
        return
    for n in range(a.start, (a.end or a.start) + 1):
        adj = {v: set(w) for v, w in build_graph(n).items()}
        r = ham_kpw.kpw_order(adj, cost="work", work_model=a.work_model)
        res = exact(n, list(r.order), adj)
        summ = summarize(res)
        summ["features"] = ordering_features(adj, list(r.order))
        with open(os.path.join(a.out, f"exact_{n}.json"), "w") as f:
            json.dump(dict(summary=summ, rows=res["rows"], order=res["order"]), f)
        print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
