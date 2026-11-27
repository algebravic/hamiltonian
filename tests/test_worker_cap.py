"""Per-step worker buffer size (doc/worker_cap_design.md): the choice of the
buffer size must never change the tables, and 'auto' must avoid spills that
'fixed' cannot avoid under a tight budget."""
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent("""
    import sys
    from hamiltonian.claude import ham_dp_c as m
    from hamiltonian.claude.ham_ordering import build_graph, best_bfs_order
    n, yml, wc = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    m.set_machine_yaml(yml)
    adj = build_graph(n); order = best_bfs_order(adj, n)
    print("PATHS", m.count_hamiltonian_paths_sm(n, order, adj, instrument=True,
                                                ram_bytes=16 << 30, worker_cap=wc))
""")


def _run(tmp_path, n, cap, wc, cap_min, env_extra=None):
    yml = tmp_path / f"m_{cap}_{cap_min}.yaml"
    yml.write_text(f"SM_NTHREADS: 3\nSM_WORKER_CAP: {cap}\nSM_WORKER_CAP_MIN: {cap_min}\n"
                   "SM_MAX_RUNS: 1024\nSM_PAR_MERGE_THRESH: 1000\n")
    env = dict(os.environ, HAM_SM_CKSUM="1", TMPDIR=str(tmp_path), **(env_extra or {}))
    p = subprocess.run([sys.executable, "-c", CHILD, str(n), str(yml), wc], cwd=ROOT,
                       env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    ck = [l for l in p.stderr.splitlines() if l.startswith("[cksum]")]
    caps = [int(x) for x in re.findall(r"cap: C=(\d+)K", p.stderr)]
    lo = [int(x) for x in re.findall(r"lo=(\d+)K", p.stderr)]
    spilled = sum(int(x) for x in re.findall(r"runs: total=\d+ spilled=(\d+)", p.stderr))
    paths = int(re.search(r"^PATHS (\d+)", p.stdout, re.M).group(1))
    return ck, paths, caps, lo, spilled


@pytest.mark.parametrize("n", [36, 41])
def test_cap_never_changes_tables(tmp_path, n):
    ref = _run(tmp_path, n, 1 << 25, "fixed", 1 << 20)
    for cap, wc, cmin in [(1 << 25, "auto", 1 << 20), (1 << 20, "auto", 1 << 10),
                          (1 << 16, "fixed", 1 << 10), (4096, "auto", 256)]:
        r = _run(tmp_path, n, cap, wc, cmin)
        assert r[1] == ref[1]
        assert r[0] == ref[0]                      # every step's digest
        if wc == "auto":
            assert all(cmin >> 10 <= c <= max(cap >> 10, 1) or c == cap >> 10 for c in r[2])
            assert all(c >= l for c, l in zip(r[2], r[3]))


def test_auto_avoids_spill_under_tight_budget(tmp_path):
    env = {"HAM_SM_MERGE_BUDGET": str(300 << 20)}
    fixed = _run(tmp_path, 41, 1 << 22, "fixed", 1 << 14, env)
    auto = _run(tmp_path, 41, 1 << 22, "auto", 1 << 14, env)
    assert fixed[1] == auto[1]
    assert fixed[4] > 0 and auto[4] == 0
