"""ordering_study.py — resumable runs for validating ordering cost models and
for comparing machines (doc/work_model_redesign.md, Phase 1 and 1b).

Three subcommands:

  candidates  Build the shared candidate orderings for each n, once, and write
              them to <data>/cands_<n>.json.  Commit these files so that every
              machine runs exactly the same orderings.
                python -m hamiltonian.claude.ordering_study candidates 58 64 \\
                    --models work_model.json data/linux/linux_work_model.json \\
                             sri_work_model.json work_m2pro.json

  run         Run every candidate (in increasing order of n, then candidate) on
              this machine, with pruning and --instrument, and write one JSON per
              run under <data>/<host>/.  Existing results are skipped, so it can
              be interrupted and restarted at any time.
                python -m hamiltonian.claude.ordering_study run 58 64 \\
                    --machine-yaml yaml/new-machine.yaml --timeout 3600

  analyze     Regret of each cost model, and per-machine cost coefficients,
              from everything found under <data>/; writes <data>/analysis.md.
                python -m hamiltonian.claude.ordering_study analyze

Each run record holds: the host fingerprint (CPU, cores, RAM, caches, compiler,
build key, SM_* settings, machine.yaml contents), the ordering, the per-step
table (states in/out, n_back, frontier size) and the per-step timings and
counters from --instrument (generate, flush, merge, introduce, contributions
written, buffer size, spills), and the totals.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time

from .ham_ordering import build_graph

DEFAULT_DATA = os.path.join("data", "ordering_study")


# ---------------------------------------------------------------- candidates
def _ordering_features(adj, order):
    n = len(order)
    pos = {v: i for i, v in enumerate(order)}
    last_s = {v: max([pos[w] for w in adj[v]], default=pos[v]) for v in order}
    F, sumf, maxf, O = [], 0, 0, 0
    for t, v in enumerate(order):
        F = [u for u in F + [v] if last_s[u] > t]
        U = set(order[t + 1:])
        sumf += len(F)
        maxf = max(maxf, len(F))
        O += sum(1 for u in F if sum(1 for x in adj[u] if x in U) == 1)
    return dict(sum_f=sumf, max_f=maxf, forced_opp=O)


def _model_tag(path):
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(path))[0])


def make_candidates(n, models, data, k=3, samples=16, seed=1):
    """Candidate orderings for G_n: the optimum under each work-model file,
    the v7 and v2 optima, the k best under v2, and `samples` uniform samples
    from the kpw DAG.  Every work-model file's predicted cost of every
    candidate is stored too, for the regret analysis."""
    from . import ham_kpw
    from .work_model import WorkModel
    path = os.path.join(data, f"cands_{n}.json")
    if os.path.exists(path):
        print(f"{path} exists; not regenerated", flush=True)
        return path
    adj = {v: set(w) for v, w in build_graph(n).items()}
    orders, seen = [], set()

    def add(tag, o):
        key = tuple(o)
        if key in seen:
            for c in orders:
                if tuple(c["order"]) == key:
                    c["also"].append(tag)
            return
        seen.add(key)
        orders.append(dict(tag=tag, also=[], order=list(o)))

    for m in models:
        r = ham_kpw.kpw_order(adj, cost="work", work_model=m, engine="dsl")
        add(f"opt-{_model_tag(m)}", r.order)
        pw = r.pw
    for c in ("v7", "v2"):
        r = ham_kpw.kpw_order(adj, cost=c, engine="dsl")
        add(f"opt-{c}", r.order)
        pw = r.pw
    pw2, extra = ham_kpw.kpw_candidates(adj, k=k, samples=samples, seed=seed, costs=(),
                                        cache_dir=None)
    for tag, o in extra:
        add(tag, o)
    g = ham_kpw.adj_to_graph(adj)
    for c in orders:
        c["features"] = _ordering_features(adj, c["order"])
        c["predicted"] = {_model_tag(m): WorkModel.load(m).predicted_work(g, c["order"], pw)
                          for m in models}
    out = dict(n=n, pw=pw, models={_model_tag(m): m for m in models},
               model_md5={_model_tag(m): hashlib.md5(open(m, "rb").read()).hexdigest()
                          for m in models},
               k=k, samples=samples, seed=seed, created=time.strftime("%Y-%m-%d %H:%M"),
               orders=orders)
    os.makedirs(data, exist_ok=True)
    json.dump(out, open(path, "w"))
    print(f"{path}: {len(orders)} distinct orderings (pw={pw})", flush=True)
    return path


# ---------------------------------------------------------------- host fingerprint
def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def host_fingerprint(machine_yaml=None):
    from . import ham_dp_c as m
    ram, thr = m._detect_ram_and_threads()
    l1, l2, l3, line = m._detect_cache_sizes()
    fp = dict(hostname=socket.gethostname(), system=platform.system(),
              machine=platform.machine(), release=platform.release(),
              python=platform.python_version(), ram_bytes=ram, threads_detected=thr,
              l1=l1, l2=l2, l3=l3, cache_line=line)
    try:
        fp["cache_sharing"] = m._detect_cache_sharing()
    except Exception:
        pass
    if fp["system"] == "Darwin":
        fp["cpu"] = _sh("sysctl -n machdep.cpu.brand_string")
        fp["perflevel0_cores"] = _sh("sysctl -n hw.perflevel0.physicalcpu")
        fp["perflevel1_cores"] = _sh("sysctl -n hw.perflevel1.physicalcpu")
        fp["physical_cores"] = _sh("sysctl -n hw.physicalcpu")
        fp["memsize"] = _sh("sysctl -n hw.memsize")
    else:
        fp["cpu"] = (_sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2")
                     or _sh("lscpu | grep -m1 'Model name' | cut -d: -f2"))
        fp["physical_cores"] = _sh("lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l")
        fp["numa_nodes"] = _sh("lscpu | grep -i 'NUMA node(s)' | awk '{print $NF}'")
    if machine_yaml and os.path.exists(machine_yaml):
        fp["machine_yaml_path"] = machine_yaml
        fp["machine_yaml"] = open(machine_yaml).read()
    return fp


# ---------------------------------------------------------------- one run
_CHILD = """
import sys, json
from hamiltonian.claude import ham_dp_c as m
from hamiltonian.claude.ham_ordering import build_graph
n, order, yml = int(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
if yml:
    m.set_machine_yaml(yml)
t0 = __import__("time").time()
r = m.count_hamiltonian_paths_sm(n, order, build_graph(n), verbose=True, instrument=True,
                                 prune="drop", worker_cap="auto")
print("RESULT", r, __import__("time").time() - t0)
"""

_STEP = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)")


def parse_log(out, err):
    """Per-step table from the -v lines and per-step counters from --instrument."""
    steps, inst, cur, pending_intro, dp_ms = {}, {}, None, None, None
    for l in err.splitlines():
        m = _STEP.match(l)
        if m:
            t = int(m[1])
            dp_ms = float(m[9])
            steps[t] = dict(step=t, vertex=int(m[2]), fs=int(m[3]), n_back=int(m[4]),
                            e_bag=int(m[5]), states_in=int(m[6]), states_out=int(m[7]),
                            step_ms=float(m[8]))
            continue
        if "[inst] introduce:" in l:
            x = re.search(r"([\d.]+)ms", l)
            pending_intro = float(x[1]) if x else None
        elif "[inst]   cap: C=" in l:
            pend_cap = dict(cap_k=int(re.search(r"C=(\d+)K", l)[1]),
                            pred_M=int(re.search(r"pred=(\d+)M", l)[1]))
            inst.setdefault("_cap", []).append(pend_cap)
        elif re.search(r"\[inst\] step=\d+ ", l):
            cur = int(re.search(r"step=(\d+)", l)[1])
            inst[cur] = dict(introduce_ms=pending_intro)
            if inst.get("_cap"):
                inst[cur].update(inst["_cap"].pop())
        elif cur is not None and "compute: max=" in l:
            x = re.findall(r"max=([\d.]+)ms mean=([\d.]+)ms", l)
            inst[cur].update(compute_max_ms=float(x[0][0]), compute_mean_ms=float(x[0][1]),
                             flush_max_ms=float(x[1][0]), flush_mean_ms=float(x[1][1]))
        elif cur is not None and "par_merge:" in l:
            inst[cur]["merge_ms"] = float(re.search(r"par_merge: ([\d.]+)ms", l)[1])
        elif cur is not None and "dedup_stages:" in l:
            inst[cur]["raw_M"] = int(re.search(r"raw=(\d+)M", l)[1])
            inst[cur]["after_run_M"] = int(re.search(r"after_run=(\d+)M", l)[1])
        elif cur is not None and "runs: total=" in l:
            inst[cur]["runs"] = int(re.search(r"total=(\d+)", l)[1])
            inst[cur]["spilled"] = int(re.search(r"spilled=(\d+)", l)[1])
            inst[cur]["spilled_GB"] = float(re.search(r"spilled=\d+ \(([\d.]+)GB\)", l)[1])
    inst.pop("_cap", None)
    for t, d in inst.items():
        if t in steps:
            steps[t].update(d)
    header = [l for l in out.splitlines() if l.startswith("# ham_dp_c:")]
    res = re.search(r"^RESULT (\d+) ([\d.]+)", out, re.M)
    rss = [float(x) for x in re.findall(r"maxrss_lifetime=([\d.]+)GB", err)]
    return dict(steps=[steps[t] for t in sorted(steps)], header=header,
                total=int(res[1]) if res else None, wall_s=float(res[2]) if res else None,
                dp_ms=dp_ms,
                peak_rss_GB=max(rss) if rss else None)


def run_all(ns, data, host, machine_yaml, timeout, known):
    outdir = os.path.join(data, host)
    os.makedirs(outdir, exist_ok=True)
    fp_path = os.path.join(outdir, "host.json")
    if not os.path.exists(fp_path):
        json.dump(host_fingerprint(machine_yaml), open(fp_path, "w"), indent=1)
    for n in ns:
        cpath = os.path.join(data, f"cands_{n}.json")
        if not os.path.exists(cpath):
            print(f"missing {cpath}: run 'candidates' first (and share the file)", flush=True)
            continue
        cands = json.load(open(cpath))
        for c in cands["orders"]:
            path = os.path.join(outdir, f"run_{n}_{c['tag']}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, "-c", _CHILD, str(n), json.dumps(c["order"]),
                                    machine_yaml or ""], capture_output=True, text=True,
                                   timeout=timeout)
                rec = parse_log(p.stdout, p.stderr)
                rec["returncode"] = p.returncode
                if p.returncode != 0:
                    rec["stderr_tail"] = p.stderr[-3000:]
            except subprocess.TimeoutExpired:
                rec = dict(timeout=True)
            rec.update(n=n, tag=c["tag"], host=host, date=time.strftime("%Y-%m-%d %H:%M"),
                       elapsed_s=time.time() - t0)
            if rec.get("total") is not None and str(n) in known:
                rec["correct"] = rec["total"] == known[str(n)]
            json.dump(rec, open(path, "w"))
            print(f"{host} n={n} {c['tag']}: total={rec.get('total')} dp={rec.get('dp_ms')}ms "
                  f"{'TIMEOUT' if rec.get('timeout') else ''}"
                  f"{'' if rec.get('correct', True) else ' WRONG COUNT'}", flush=True)


# ---------------------------------------------------------------- analysis
def _spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for q in range(i, j + 1):
                r[o[q]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rank(a), rank(b); n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def _work(rec):
    """W = Σ_t s_t · 2^nb_t (the quantity kpw minimises) and Σ states."""
    st = rec["steps"]
    return (sum(s["states_in"] * 2 ** s["n_back"] for s in st), sum(s["states_out"] for s in st))


def _lstsq(X, y):
    """Least squares without numpy dependence on shape quirks."""
    import numpy as np
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - pred) ** 2).sum()) / ss if ss else float("nan")
    return beta, r2


def analyze(data):
    lines = ["# Ordering study: analysis", ""]
    cands = {int(re.search(r"cands_(\d+)", p)[1]): json.load(open(p))
             for p in glob.glob(os.path.join(data, "cands_*.json"))}
    hosts = sorted(d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d)))
    runs = {}
    for h in hosts:
        for p in glob.glob(os.path.join(data, h, "run_*.json")):
            r = json.load(open(p))
            if r.get("timeout") or r.get("total") is None:
                continue
            runs[(h, r["n"], r["tag"])] = r
    # --- regret of each model (state counts are machine-independent: use any host)
    lines += ["## Regret of each cost model (Σ states with pruning; machine-independent)", "",
              "regret = actual of the model's choice / best actual among the candidates; "
              "'opt' = the model's exact DAG optimum, 'pick' = its best-predicted candidate.", ""]
    models = sorted({m for c in cands.values() for m in c["models"]})
    lines.append("| n | candidates run | best Σ states | " + " | ".join(
        f"{m}: opt / pick / ρ_S" for m in models) + " |")
    lines.append("|---" * (3 + len(models)) + "|")
    for n in sorted(cands):
        c = cands[n]
        act = {}
        for o in c["orders"]:
            r = next((runs[k] for k in runs if k[1] == n and k[2] == o["tag"]), None)
            if r:
                act[o["tag"]] = _work(r)[1]
        if len(act) < 3:
            continue
        best = min(act.values())
        cells = []
        for m in models:
            optag = f"opt-{m}"
            opt = next((o["tag"] for o in c["orders"] if o["tag"] == optag or optag in o["also"]),
                       None)
            ran = [o for o in c["orders"] if o["tag"] in act and m in o["predicted"]]
            pick = min(ran, key=lambda o: o["predicted"][m])["tag"] if ran else None
            rs = _spearman([o["predicted"][m] for o in ran], [act[o["tag"]] for o in ran])
            f = lambda t: f"{act[t] / best:.3f}" if t in act else "—"
            cells.append(f"{f(opt) if opt else '—'} / {f(pick) if pick else '—'} / {rs:.2f}")
        lines.append(f"| {n} | {len(act)} | {best:,} | " + " | ".join(cells) + " |")
    # --- per-machine cost coefficients
    lines += ["", "## Per-machine cost coefficients (per step, steps with ≥ 1 M input states)", "",
              "generate_ms ≈ a·(s_in·masks(nb))/1e6 + b·written/1e6;  flush_ms ≈ c·written/1e6;  "
              "merge_ms ≈ d·after_run/1e6  (ns per unit = coefficient)", "",
              "| host | cpu | threads | RAM GB | steps | a (gen/mask) | b (gen/out) | R² gen | "
              "c (flush/out) | R² | d (merge/run entry) | R² | ns/state overall |",
              "|---" * 13 + "|"]
    for h in hosts:
        fp = json.load(open(os.path.join(data, h, "host.json"))) if os.path.exists(
            os.path.join(data, h, "host.json")) else {}
        X, yg, Xf, yf, Xm, ym, tot_ms, tot_states = [], [], [], [], [], [], 0.0, 0
        for (hh, n, tag), r in runs.items():
            if hh != h:
                continue
            for s in r["steps"]:
                if s.get("compute_max_ms") is None or s["states_in"] < 1_000_000:
                    continue
                nb = s["n_back"]; masks = 1 + nb + nb * (nb - 1) // 2
                w = (s.get("raw_M") or 0) * 1e6
                X.append([s["states_in"] * masks / 1e6, w / 1e6]); yg.append(s["compute_max_ms"])
                Xf.append([w / 1e6]); yf.append(s["flush_max_ms"])
                Xm.append([(s.get("after_run_M") or 0)]); ym.append(s.get("merge_ms") or 0)
            tot_ms += r["dp_ms"] or 0
            tot_states += _work(r)[1]
        if len(X) < 5:
            continue
        (a, b), r2g = _lstsq(X, yg)
        (c,), r2f = _lstsq(Xf, yf)
        (d,), r2m = _lstsq(Xm, ym)
        thr = re.search(r"SM_NTHREADS=(\d+)", " ".join(
            next((r["header"] for k, r in runs.items() if k[0] == h), [""])))
        lines.append(f"| {h} | {fp.get('cpu', '?')} | {thr[1] if thr else '?'} | "
                     f"{(fp.get('ram_bytes') or 0) / 2**30:.0f} | {len(X)} | {a * 1e3:.1f} | "
                     f"{b * 1e3:.1f} | {r2g:.2f} | {c * 1e3:.1f} | {r2f:.2f} | {d * 1e3:.1f} | "
                     f"{r2m:.2f} | {tot_ms * 1e6 / max(tot_states, 1):.1f} |")
    lines += ["", "Units: a, b, c, d are ms per million units, i.e. ns per unit. "
              "Times are the slowest worker's (generate, flush) or the parallel merge, "
              "so they are wall-clock per step, not CPU time."]
    out = "\n".join(lines) + "\n"
    open(os.path.join(data, "analysis.md"), "w").write(out)
    print(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["candidates", "run", "analyze"])
    ap.add_argument("start", type=int, nargs="?")
    ap.add_argument("end", type=int, nargs="?")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--models", nargs="*", default=[],
                    help="(candidates) work-model JSON files to compare")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--host", default=None,
                    help="(run) name of this machine's result directory (default: hostname)")
    ap.add_argument("--machine-yaml", default=None)
    ap.add_argument("--timeout", type=float, default=3600.0, help="(run) seconds per run")
    a = ap.parse_args(argv)
    ns = range(a.start, (a.end or a.start) + 1) if a.start else []
    if a.mode == "candidates":
        for n in ns:
            make_candidates(n, a.models, a.data, k=a.k, samples=a.samples, seed=a.seed)
    elif a.mode == "run":
        known = {}
        b = os.path.join("data", "b071983.txt")
        if os.path.exists(b):
            known = {x.split()[0]: int(x.split()[1]) for x in open(b) if x.strip()}
        host = a.host or socket.gethostname().split(".")[0]
        run_all(ns, a.data, host, a.machine_yaml, a.timeout, known)
    else:
        analyze(a.data)


if __name__ == "__main__":
    main()
