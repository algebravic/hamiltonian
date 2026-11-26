"""Sort-merge DP: the RAM-path merge variants must give identical tables.

Runs the sort-merge frontier DP (hamiltonian.claude.ham_dp_c) in a child
process per configuration, with a tiny SM_WORKER_CAP so that workers
produce several runs per step, and HAM_SM_CKSUM=1 so the C code prints an
order-independent digest of the state table after every step (plus a
strict-sortedness check).  The configurations are

  a0       per-worker bucket-parallel merge + P-way merge (A0)
  a1       global memory-budgeted merge with streaming release (A1)
  a1_tight A1 with a small budget (fewer threads / small release granule,
           falling back to A0 on steps that do not fit)
  spill    runs written to the spill file (partly / all), merged by A1
           from mixed RAM and file streams
  ext      the old anonymous-mapping layout (HAM_SM_LAYOUT=ext)
  lowram   nominal RAM so small that the spill rule sends every run to disk

and the test checks that the per-step digests and the final path counts
agree across all of them, and with the EH backend's count.
"""
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
    n, yml, reserve = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
    m.set_machine_yaml(yml)
    adj = build_graph(n); order = best_bfs_order(adj, n)
    # Fixed nominal RAM so the RAM/ext decisions do not depend on the host
    # (the tables here are a few MB; nothing near this is allocated).
    kw = {"ram_bytes": 16 << 30}
    if reserve > 0:
        kw["mem_reserve_gb"] = reserve
    r = m.count_hamiltonian_paths_sm(n, order, adj, instrument=True, **kw)
    print("PATHS", r)
    if len(sys.argv) > 4:
        print("EH", m.count_hamiltonian_paths_c(n, order, adj))
""")

YAML = """\
SM_NTHREADS: 3
SM_WORKER_CAP: 8192
SM_RBUF_SIZE: 32
SM_BB_BUF: 16
SM_EXT_STREAM_BUF: 1024
SM_SLC_BYTES: 4194304
SM_PAR_MERGE_THRESH: 1000
SM_STEAL_FACTOR: 32
"""


def _run(tmp_path, n, env_extra, reserve=0.0, with_eh=False):
    yml = tmp_path / "machine.yaml"
    yml.write_text(YAML)
    env = dict(os.environ, HAM_SM_CKSUM="1", **env_extra)
    args = [sys.executable, "-c", CHILD, str(n), str(yml), str(reserve)]
    if with_eh:
        args.append("eh")
    p = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    ck = [l for l in p.stderr.splitlines() if l.startswith("[cksum]")]
    spilled = sum(int(x) for x in re.findall(r"runs: total=\d+ spilled=(\d+)", p.stderr))
    paths = int(re.search(r"^PATHS (\d+)", p.stdout, re.M).group(1))
    eh = re.search(r"^EH (\d+)", p.stdout, re.M)
    modes = re.findall(r"ram_merge: mode=(\w+)", p.stderr)
    return ck, paths, (int(eh.group(1)) if eh else None), modes, spilled


@pytest.mark.parametrize("n", [36, 41])
def test_merge_variants_identical(tmp_path, n):
    a0 = _run(tmp_path, n, {"HAM_SM_MERGE": "a0"}, with_eh=True)
    a1 = _run(tmp_path, n, {"HAM_SM_MERGE": "a1"})
    tight = _run(tmp_path, n, {"HAM_SM_MERGE_BUDGET": "3000000"})
    part = _run(tmp_path, n, {"HAM_SM_SPILL_LIMIT": "300000", "TMPDIR": str(tmp_path)})
    allsp = _run(tmp_path, n, {"HAM_SM_SPILL_LIMIT": "0"})
    ext = _run(tmp_path, n, {"HAM_SM_LAYOUT": "ext"}, reserve=1e6)
    lowram = _run(tmp_path, n, {}, reserve=1e6)

    assert a0[1] == a0[2], "sort-merge count differs from EH backend"
    assert set(a0[3]) <= {"a0"} and "a1" in a1[3]
    assert part[4] > 0 and allsp[4] > part[4] and lowram[4] > 0
    assert a1[4] == 0
    for other in (a1, tight, part, allsp, ext, lowram):
        assert other[1] == a0[1]
        assert other[0] == a0[0]                  # every step's digest
    assert all(l.endswith("sorted=1") for l in a1[0])
