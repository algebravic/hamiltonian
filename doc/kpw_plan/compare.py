import sys, json, re
sys.path.insert(0, "/home/claude/kint")
from hamiltonian.claude.ham_ordering import build_graph, _dp_cost, frontier_stats
from hamiltonian.claude import sa_cost
print("MLP model:", sa_cost.model_info())
sa = {int(k): v for k, v in json.load(open("/home/claude/kint/sa_orders.json")).items()}
dagopt = {}
for line in open("/home/claude/kint/v7_runs.log"):
    m = re.match(r"n=(\d+) pw=(\d+)", line)
    if m: cur = (int(m[1]), int(m[2]))
    if line.startswith("order "): dagopt[cur[0]] = (cur[1], [int(x) for x in line.split()[1:]])
RAM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
print(f"ram_gb={RAM}")
print(f"{'n':>3} {'pw':>3} | {'V7 SA':>9} {'V7 DAG':>9} {'ratio':>6} | {'MLP SA':>8} {'MLP DAG':>8} {'diff':>6} | prof SA/DAG")
for n in sorted(dagopt):
    pw, d = dagopt[n]; s = sa[n]; adj = build_graph(n)
    v_s, v_d = _dp_cost(adj, s, pw), _dp_cost(adj, d, pw)
    m_s, m_d = sa_cost.sa_cost_fn(s, adj, n, ram_gb=RAM), sa_cost.sa_cost_fn(d, adj, n, ram_gb=RAM)
    print(f"{n:3d} {pw:3d} | {v_s:9.4g} {v_d:9.4g} {v_d/v_s:6.3f} | {m_s:8.3f} {m_d:8.3f} {m_d-m_s:+6.3f} | {frontier_stats(adj,s)[1]}/{frontier_stats(adj,d)[1]}")
