"""Predicted-work model W = sum_t s_t 2^{nb_t}, s_{t+1} = s_t exp(g(features_t)),
with g linear in step-local features, fitted with n held out.  Exact optimum over the DAG."""
import sys, json, math, re
import numpy as np
sys.path.insert(0, "/home/claude/kint/kobayashi-pw/src"); sys.path.insert(0, "/home/claude/kint"); sys.path.insert(0, "/home/claude/kint/scripts")
from kobayashi_pw.orders import OrderDAG, optimize, tau_features, BELL
from fit_local import X, y, grp, feat_vec, graph       # re-fits on import (prints)
def beta_without(n):
    m = grp != n
    return np.linalg.lstsq(X[m], y[m], rcond=None)[0]
def make_model(beta, n):
    def model(f, pw):
        g = float(np.dot(beta, feat_vec(f, n)))
        e = 2.0 ** f.n_back
        return 0.0, e, math.exp(g) / e
    return model
def pred_W(g, order, model, pw):
    cost, p = 0.0, 1.0
    for f in tau_features(g, order):
        a, e, k = model(f, pw); cost += a + p * e; p = max(p * e * k, 1.0)
    return cost
if __name__ == "__main__":
    sa = {int(k): v for k, v in json.load(open("/home/claude/kint/sa_orders.json")).items()}
    v7 = {}
    for line in open("/home/claude/kint/v7_runs.log"):
        m = re.match(r"n=(\d+) pw=(\d+)", line)
        if m: cur = int(m[1])
        if line.startswith("order "): v7[cur] = [int(x) for x in line.split()[1:]]
    out = {}
    for n in [int(a) for a in sys.argv[1:]]:
        g = graph(n); b = beta_without(n); model = make_model(b, n)
        dag = OrderDAG.load(f"/home/claude/kint/sq/sq{n}.dag")
        dag.g = g
        r = optimize(dag, model)
        print(f"n={n} predW: SA={pred_W(g, sa[n], model, dag.pw):.3e} V7opt={pred_W(g, v7[n], model, dag.pw):.3e} "
              f"Wopt={r.cost:.3e} labels={r.max_labels}", flush=True)
        out[n] = r.tau
    json.dump(out, open(f"/home/claude/kint/wopt_{'_'.join(sys.argv[1:])}.json", "w"))
