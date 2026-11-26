"""Sort-merge DP: exact counts via the CRT when 64-bit state counts wrap.

The DP only adds counts, so with wrapping arithmetic it is exact mod 2^64;
count_hamiltonian_paths_sm recovers the full value from replays mod primes.
To exercise this at small n the test hook _word_bits=b emulates b-bit
counts, so wraps happen early.  Checked:

  * exact paths and cycles for several word sizes, with the snapshot in
    RAM (rolling snapshots: replay from the wrap step) and on disk
    (crt_snap_bytes=0: the anchor is written to the spill directory and
    the replay starts at the first step where a wrap is possible);
  * the no-wrap test: no count wraps at a step with U_s < 2^b;
  * on_overflow='residue' raises SMOverflow with the right residue;
  * three moduli (small primes patched in) combine correctly;
  * the ext path and later calls are unaffected (flags are reset);
  * the Bregman bound and the prime list.
"""
import math

import pytest

from hamiltonian.claude import ham_dp_c as m
from hamiltonian.claude.ham_ordering import best_bfs_order, build_graph

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
RAM = 16 << 30
N = 41


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    yml = tmp_path_factory.mktemp("crt") / "machine.yaml"
    yml.write_text(YAML)
    m.set_machine_yaml(str(yml))
    adj = build_graph(N)
    order = best_bfs_order(adj, N)
    ref = m.count_hamiltonian_paths_sm(N, order, adj, ram_bytes=RAM, count_cycles=True)
    yield adj, order, ref
    m.set_machine_yaml(None)


@pytest.mark.parametrize("bits", [20, 12])
@pytest.mark.parametrize("snap", [1 << 30, 0])
def test_crt_exact(setup, bits, snap):
    adj, order, ref = setup
    r = m.count_hamiltonian_paths_sm(N, order, adj, ram_bytes=RAM, count_cycles=True,
                                     _word_bits=bits, crt_snap_bytes=snap)
    assert r == ref


def test_residue_mode(setup):
    adj, order, ref = setup
    with pytest.raises(m.SMOverflow) as ei:
        m.count_hamiltonian_paths_sm(N, order, adj, ram_bytes=RAM,
                                     _word_bits=16, on_overflow="residue")
    assert ei.value.residue == ref[0] % (1 << 16)
    assert ei.value.step >= 0
    # a normal call afterwards is unaffected
    assert m.count_hamiltonian_paths_sm(N, order, adj, ram_bytes=RAM) == ref[0]


def test_three_small_moduli(setup, monkeypatch):
    adj, order, ref = setup
    small = [65521, 65519, 65497, 65479]
    monkeypatch.setattr(m, "crt_primes", lambda k: small[:k])
    for snap in (1 << 30, 0):
        r = m.count_hamiltonian_paths_sm(N, order, adj, ram_bytes=RAM, count_cycles=True,
                                         _word_bits=12, crt_snap_bytes=snap)
        assert r == ref


def test_ext_path_with_wraps(setup):
    adj, order, ref = setup
    r = m.count_hamiltonian_paths_sm(N, order, adj, ram_bytes=RAM, mem_reserve_gb=1e6,
                                     _word_bits=20)
    assert r == ref[0]


def test_bound_and_primes():
    ps = m.crt_primes(3)
    assert ps[0] == (1 << 61) - 1 and all(m._is_prime(p) for p in ps)
    assert all(not m._is_prime(q) for q in range(ps[1] + 2, ps[0], 2))
    # Bregman bound matches hamiltonian/bounds/bregman.py and exceeds known H(n)
    from hamiltonian.bounds.bregman import log_upper_bound
    for n, h in ((41, None), (74, 2176025661130535741)):
        b = m.bregman_log2_bound(n, build_graph(n))
        assert abs(b - log_upper_bound(n) / math.log(2)) < 1e-9
        if h:
            assert math.log2(h) < b
    assert m.crt_combine([5, 5 % 7, 5 % 11], [1 << 64, 7, 11]) == 5


def test_no_wrap_before_risky_step(setup):
    """Every count after step s is <= U_s = 2^n_back * (sum of input counts),
    so the first wrap cannot precede the first step with U_s >= 2^b."""
    import re, subprocess, sys, textwrap
    adj, order, ref = setup
    for bits in (12, 16, 20):
        code = textwrap.dedent(f"""
            from hamiltonian.claude import ham_dp_c as m
            from hamiltonian.claude.ham_ordering import build_graph, best_bfs_order
            m.set_machine_yaml({str(m._MACHINE_YAML_PATH)!r})
            adj = build_graph({N}); order = best_bfs_order(adj, {N})
            m.count_hamiltonian_paths_sm({N}, order, adj, ram_bytes={RAM},
                                         _word_bits={bits}, instrument=True)
        """)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
        assert p.returncode == 0, p.stderr[-2000:]
        steps = [int(x) for x in re.findall(r"\[inst\] step=(\d+) ", p.stderr)]
        u = [int(x) for x in re.findall(r"crt: log2U=(-?\d+)", p.stderr)]
        wrap = int(re.search(r"first at step (\d+)", p.stderr).group(1))
        first_risky = next(i for i, l in enumerate(u) if l >= bits)
        assert len(u) >= wrap
        assert wrap >= first_risky, (bits, wrap, first_risky)
