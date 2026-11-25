"""Fit a step-local predictor of log(states_out/states_in) from the logged runs,
using features computable on DAG arcs (orders.StepFeatures).  Leave-one-n-out."""
import sys, json, math
import numpy as np
sys.path.insert(0, "/home/claude/kint/kobayashi-pw/src"); sys.path.insert(0, "/home/claude/kint")
from kobayashi_pw.orders import tau_features, BELL
from kobayashi_pw.io import from_edges
from hamiltonian.claude.ham_ordering import build_graph
L = "/mnt/user-data/uploads/hamiltonian/data/linux/"
files = {n: f"profile_{n}linux.txt" for n in range(57, 75)}; files[75] = "profile_75log.txt"
def graph(n):
    adj = build_graph(n)
    return from_edges([(u, v) for u in adj for v in adj[u] if u < v], vertices=range(1, n + 1))
def feat_vec(f, n):
    lb = sum(math.log(BELL[min(s, 20)]) for s in f.comp_sizes)
    return [1.0, f.n_back, f.n_elim, f.fw, f.fw_before, f.e_bag, lb, f.n_back * f.fw, f.n_elim * f.fw_before,
            f.n_back ** 2, f.step / (n - 1)]
NAMES = ["1","nb","nelim","fw","fw0","ebag","logBellComp","nb*fw","nelim*fw0","nb^2","step/n"]
X, y, grp = [], [], []
for n, fn in files.items():
    rows = []
    for line in open(L + fn):
        t = line.split()
        if len(t) >= 10 and t[0].isdigit() and t[1].isdigit() and t[-1] in ("[sm]", "[ext]"):
            rows.append((int(t[1]), int(t[5]), int(t[6])))
    g = graph(n); order = [r[0] for r in rows]
    feats = tau_features(g, order)
    for f, (v, si, so) in zip(feats, rows):
        assert g.labels[f.v] == v
        if si >= 1000 and so > 0:
            X.append(feat_vec(f, n)); y.append(math.log(so / si)); grp.append(n)
X, y, grp = np.array(X), np.array(y), np.array(grp)
beta = np.linalg.lstsq(X, y, rcond=None)[0]
r2 = 1 - np.sum((X @ beta - y) ** 2) / np.sum((y - y.mean()) ** 2)
pred = np.zeros_like(y)
for n in np.unique(grp):
    m = grp == n
    b = np.linalg.lstsq(X[~m], y[~m], rcond=None)[0]; pred[m] = X[m] @ b
r2cv = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"obs={len(y)}  R2={r2:.3f}  leave-one-n-out R2={r2cv:.3f}  rmse={np.sqrt(np.mean((pred-y)**2)):.3f}")
for nm, b in zip(NAMES, beta): print(f"  {nm:12s} {b:+.4f}")
json.dump({"names": NAMES, "beta": beta.tolist()}, open("/home/claude/kint/local_model.json", "w"))
