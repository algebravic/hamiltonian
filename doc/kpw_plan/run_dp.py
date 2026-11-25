"""Run the sort-merge DP on one ordering; print time, count, and the actual work."""
import sys, json, time, re
sys.path.insert(0, "/home/claude/kint")
from hamiltonian.claude.ham_ordering import build_graph
from hamiltonian.claude import ham_dp_c
ham_dp_c.set_machine_yaml("/home/claude/kint/machine_container.yaml")
n, which = int(sys.argv[1]), sys.argv[2]
order = [int(x) for x in sys.argv[3].split(",")]
adj = build_graph(n)
t = time.time()
c = ham_dp_c.count_hamiltonian_paths_sm(n, order, adj, verbose=True)
print(f"RESULT n={n} which={which} dp={time.time()-t:.2f}s count={c}", flush=True)
