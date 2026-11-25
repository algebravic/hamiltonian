import sys, time, os
sys.path.insert(0, "/home/claude/kint/kobayashi-pw/src")
sys.path.insert(0, "/home/claude/kint")
from kobayashi_pw.orders import OrderDAG, optimize, make_model_v7, evaluate
from hamiltonian.claude.ham_ordering import _dp_cost, build_graph, frontier_stats
n = int(sys.argv[1])
t0 = time.time(); dag = OrderDAG.load(f"/home/claude/kint/sq/sq{n}.dag"); t1 = time.time()
r = optimize(dag, make_model_v7()); t2 = time.time()
adj = build_graph(n)
c = _dp_cost(adj, r.tau, dag.pw)
mx, pr = frontier_stats(adj, r.tau)
print(f"n={n} pw={dag.pw} nodes={len(dag)} orderings={dag.num_orderings():.2e} load={t1-t0:.1f}s opt={t2-t1:.1f}s "
      f"v7={r.cost:.6g} _dp_cost={c:.6g} max_fw={mx} profile={pr} max_labels={r.max_labels}", flush=True)
print("order", " ".join(map(str, r.tau)))
