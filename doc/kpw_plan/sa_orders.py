"""Extract the orderings used in the Linux runs from the step column of the logs,
and check them against the logged frontier sizes."""
import sys, re, json
sys.path.insert(0, "/home/claude/kint")
from hamiltonian.claude.ham_ordering import build_graph, frontier_width
L = "/mnt/user-data/uploads/hamiltonian/data/linux/"
files = {n: f"profile_{n}linux.txt" for n in range(57, 75)}
files[75] = "profile_75log.txt"
out = {}
for n, f in files.items():
    order, fs = [], []
    for line in open(L + f):
        t = line.split()
        if len(t) >= 10 and t[0].isdigit() and t[1].isdigit() and t[-1] in ("[sm]", "[ext]"):
            order.append(int(t[1])); fs.append(int(t[2]))
    assert sorted(order) == list(range(1, n + 1)), (n, len(order))
    fw = frontier_width(build_graph(n), order)
    ok = fw == fs
    out[n] = order
    print(n, "steps", len(order), "fs matches log:", ok, "max", max(fs))
json.dump(out, open("/home/claude/kint/sa_orders.json", "w"))
