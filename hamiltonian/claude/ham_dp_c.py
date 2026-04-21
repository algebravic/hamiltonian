"""
ham_dp_c.py  (EH edition)
--------------------------
C-accelerated frontier DP for counting Hamiltonian paths in G_n.

Hash table redesign: Extendible Hashing (Fagin et al. 1979)
------------------------------------------------------------
Previous versions used a linear-probing hash table pre-allocated at 1.5×
the input state count (slot_cap-bounded).  This caused two problems:

  1. Pre-allocation mismatch: nxt was sized for 1.5× si even when the
     actual output was much smaller (e.g. step 40 n=61: 19.2M → 4.2M
     states, 6.2% fill, yet a 1.61 GB table was allocated).

  2. 3-table resize spike: when LP's nxt grew past the load factor during
     a sweep, it doubled (old_nxt + new_nxt + curr = 3 tables simultaneously),
     causing a transient 2–3× memory spike that pushed macOS into swap.

Extendible hashing fixes both:

  - Memory ∝ actual entries, not estimated capacity.
  - No resize spike: bucket splits are incremental (one new bucket per split).
  - Write buffer is no longer needed: EH already groups entries by hash prefix.
  - --load-factor tuning is no longer needed.

EH structure
------------
  EHBkt: 128 key/value pairs + cnt + local_depth = 3080 bytes.
         128 × 24B = 3072B of data fits within ~2 L1 cache lines for scan.

  EHSlab: 4096 contiguous EHBkt objects (~12.6 MB), bump-allocated.
          Pool is rewound (not freed) between steps.

  EHT: directory of 2^gd bucket pointers.
       Starts at gd=4 (16 root buckets).
       Doubles the directory when a bucket at full local depth fills.

  eh_insert: search bucket (L1-resident) for existing key → accumulate;
             otherwise insert; if bucket full → split + retry.
  eh_reset:  rewind pool (O(n_slabs)), rebuild tiny directory (O(16)).
  EH_FOR:    iterate unique buckets via pool traversal (not via directory).

Public API
----------
count_hamiltonian_paths_c(n, order, adj)  ->  int
_get_lib()                                ->  (lib, ffi)
"""

import os, hashlib, subprocess, tempfile, cffi

import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).parent
C_SOURCE = (_HERE / "ham_dp_c.c").read_text()

# ---------------------------------------------------------------------------
# Cache geometry detection and compile-time constant derivation
# ---------------------------------------------------------------------------

def _detect_cache_sizes():
    """Return (l1d_bytes, l2_bytes, l3_bytes, cacheline_bytes).

    Tries:
      macOS  — sysctl hw.l1dcachesize / hw.l2cachesize / hw.l3cachesize
               Apple Silicon exposes hw.l3cachesize directly; older Macs may
               not have L3, so we fall back to hw.cachesize (total).
      Linux  — /sys/devices/system/cpu/cpu0/cache/index*/  (most reliable)
               then /proc/cpuinfo 'cache size' line as a fallback.
    Falls back to conservative x86 defaults on failure.
    """
    import platform, re
    defaults = (32 << 10, 512 << 10, 8 << 20, 64)   # 32KB / 512KB / 8MB / 64B

    def sysctl_int(key):
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        try: return int(r.stdout.strip()) if r.returncode == 0 else None
        except ValueError: return None

    sys_ = platform.system()
    if sys_ == "Darwin":
        l1  = sysctl_int("hw.l1dcachesize")
        l2  = sysctl_int("hw.l2cachesize")
        l3  = sysctl_int("hw.l3cachesize") or sysctl_int("hw.cachesize")
        cl  = sysctl_int("hw.cachelinesize")
        return (l1 or defaults[0], l2 or defaults[1],
                l3 or defaults[2], cl or defaults[3])

    elif sys_ == "Linux":
        l1d = l2 = l3 = None
        cl = 64
        for idx in range(8):
            base = f"/sys/devices/system/cpu/cpu0/cache/index{idx}"
            if not os.path.exists(base):
                break
            try:
                level  = int(open(f"{base}/level").read())
                ctype  = open(f"{base}/type").read().strip()
                s_str  = open(f"{base}/size").read().strip()
                cl     = int(open(f"{base}/coherency_line_size").read())
                mult   = (1 << 20) if s_str.endswith("M") else (1 << 10)
                size   = int(s_str.rstrip("KMG")) * mult
                if level == 1 and "Data" in ctype: l1d = size
                elif level == 2:                   l2  = size
                elif level == 3:                   l3  = size
            except Exception:
                continue
        if l3 is None:                              # /proc/cpuinfo fallback
            try:
                for line in open("/proc/cpuinfo"):
                    m = re.search(r"cache size\s*:\s*(\d+)\s*KB", line)
                    if m: l3 = int(m.group(1)) << 10; break
            except Exception:
                pass
        return (l1d or defaults[0], l2 or defaults[1],
                l3 or defaults[2], cl)

    return defaults


def _detect_ram_and_threads():
    """Return (physical_ram_bytes, n_perf_cores).

    RAM:
      macOS  — sysctl hw.memsize
      Linux  — /proc/meminfo MemTotal
    Threads:
      macOS  — sysctl hw.perflevel0.logicalcpu  (P-cores only on Apple Silicon)
               falls back to hw.logicalcpu
      Linux  — nproc / os.cpu_count()
    Falls back to 8 GB RAM, 6 threads on failure.
    """
    import platform

    def sysctl_int(key):
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        try: return int(r.stdout.strip()) if r.returncode == 0 else None
        except ValueError: return None

    sys_ = platform.system()
    ram = None
    threads = None

    if sys_ == "Darwin":
        ram     = sysctl_int("hw.memsize")
        threads = (sysctl_int("hw.perflevel0.logicalcpu")   # P-cores (Apple Silicon)
                   or sysctl_int("hw.logicalcpu"))

    elif sys_ == "Linux":
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) * 1024
                    break
        except Exception:
            pass
        threads = os.cpu_count()

    ram     = ram     or (8 << 30)   # 8 GB fallback
    threads = threads or 6
    return ram, threads


def _derive_build_constants(l1d: int, l2: int, l3: int, cl: int) -> dict:
    """Derive EH / rsort / SM compile-time constants from hardware geometry.

    EH_BKT_CAP:
        A bucket must fit comfortably in L1 so the sequential key scan is
        cache-resident.  32 entries × 24 B = 768 B (12 cache lines at 64 B)
        is well within any L1 ≥ 16 KB.  We use 64 only if L1 ≥ 64 KB
        (uncommon, but some server parts have this).

    RSORT_THRESH:
        Activate the radix-sort buffer when the EH bucket pool would exceed
        L3 — at that point random eh_insert calls become DRAM-bound.
        pool_bytes ≈ states × (bkt_cap × 24 + 8) / (bkt_cap × 0.75)
        Solve for states: thresh = L3 / bytes_per_state, rounded to pow2.

    RSORT_BITS (→ RSORT_SLOTS = 2^RSORT_BITS):
        Each rsort flush processes all entries for a contiguous range of
        2^(gd - RSORT_BITS) EH buckets.  We want that flush region to fit
        in L2 with ~16 regions simultaneously warm.
        Sizing at the 30 M-state design point (covers n=58 peak; conservative
        for n=61):
            flush_bytes = (30M / fill / 2^bits) × bkt_bytes
            target      = L2 / 16
        Clamped to [8, 12] (256 … 4096 slots) to avoid excessive malloc.

    SM_NTHREADS:
        Number of SM worker threads.  Use all P-cores (performance cores on
        Apple Silicon, all cores on other hardware).

    SM_WORKER_CAP:
        Per-worker output buffer cap (entries).  Bounds peak RAM to:
            curr_tab + SM_NTHREADS × 2 × SM_WORKER_CAP × 24B + headroom
        Solve for cap given available RAM, leaving OS_HEADROOM_GB free:
            cap = (ram - os_headroom - curr_max) / (nthreads × 2 × 24)
        curr_max is conservatively sized for a 700M-state table (n=61 peak).
        Result is rounded down to a power-of-2 multiple of 1M entries and
        clamped to [8M, 256M].
    """
    import math
    ram, n_threads = _detect_ram_and_threads()

    # ── EH constants ──────────────────────────────────────────────────
    bkt_cap = 64 if l1d >= (64 << 10) else 32
    bkt_bytes = bkt_cap * 24 + 8
    bps = bkt_bytes / (bkt_cap * 0.75)           # bytes per state in pool

    thresh_raw = int(l3 / bps)
    thresh = 1 << max(0, thresh_raw.bit_length() - 1)   # round down to pow2

    design_peak  = 30_000_000
    total_pool   = design_peak / (bkt_cap * 0.75) * bkt_bytes
    flush_target = max(l2 // 16, 64 << 10)        # L2/16, floor 64 KB
    bits_raw     = math.ceil(math.log2(total_pool / flush_target))
    bits         = max(8, min(12, bits_raw))

    # ── SM constants ──────────────────────────────────────────────────
    SM_ENTRY     = 24          # bytes per SMEntry
    OS_HEADROOM  = 4 << 30     # 4 GB reserved for OS + misc

    # During the sweep compute phase the peak memory footprint is:
    #   curr (largest sweep input ≈ 553M entries for n=61 step 38) = 13.3 GB
    #   P × buf  (SM_WORKER_CAP entries each)
    #   P × tmp  (SM_WORKER_CAP entries each, freed before sm_merge_runs)
    #   run data accumulating ≈ 2 × (cap/dedup) × P entries at peak
    #     (two completed runs per worker before the third flush; dedup≈4×)
    #
    # Total ≈ curr + P × cap × 24 × (2 + 2 + 0.5)  = curr + P×cap×24×2.5
    # Solve for cap given a target of (RAM - OS_HEADROOM - 2 GB slack):
    curr_max_bytes = 553_000_000 * SM_ENTRY   # 13.3 GB

    target = ram - OS_HEADROOM - (2 << 30)    # 30 GB on 36 GB machine
    cap_bytes = max(0, int((target - curr_max_bytes) / (n_threads * 2.5 * SM_ENTRY)))
    # Round down to nearest 1M-entry boundary, clamp to [8M, 128M]
    cap_m = max(8, min(128, cap_bytes // (1024 * 1024)))
    sm_worker_cap = cap_m * 1024 * 1024

    # ── SM buffer tuning constants ───────────────────────────────────────
    # Load from machine.yaml (written by tune_params.py).
    # Search order:
    #   1. _MACHINE_YAML_PATH if set via set_machine_yaml() / --machine-yaml
    #   2. machine.yaml next to this script
    #   3. machine.yaml in the current working directory
    # Falls back to analytical defaults if no file is found.
    import yaml as _yaml, pathlib as _pl
    _here = _pl.Path(__file__).parent
    _candidates = []
    if _MACHINE_YAML_PATH is not None:
        _candidates.append(_pl.Path(_MACHINE_YAML_PATH))
    _candidates.append(_here / "machine.yaml")
    _candidates.append(_pl.Path.cwd() / "machine.yaml")
    _tuned = {}
    _yaml_used = None
    for _yaml_path in _candidates:
        if _yaml_path.exists():
            try:
                with open(_yaml_path) as _f:
                    _tuned = _yaml.safe_load(_f) or {}
                if _tuned:
                    _yaml_used = _yaml_path
                    break   # use first file that loads non-empty
            except Exception:
                pass  # try next candidate

    def _t(key, default):
        """Return tuned value from YAML or the analytical default."""
        return int(_tuned.get(key, default))

    # Defaults are derived analytically from hardware geometry.
    # SM_RBUF_SIZE: fill L2 with RadixBuf (256 buckets × size × 24B ≤ L2)
    rbuf_default  = 1
    while (rbuf_default * 2) * 256 * 24 <= l2: rbuf_default *= 2
    rbuf_default  = min(rbuf_default, 128)   # practical cap

    # SM_BB_BUF: fill L1 with block-merge buffers (P × size × 24B ≤ L1)
    bb_default    = 1
    while (bb_default * 2) * n_threads * 24 <= l1d: bb_default *= 2
    bb_default    = min(bb_default, 512)

    # SM_EXT_STREAM_BUF: floor; dynamic formula in C overrides per step
    ext_default   = max(256, l3 // (100 * 24))   # rough: 1% of SLC each

    # SM_SLC_BYTES: the SLC (L3) size itself
    slc_bytes     = l3

    # Only the two compile-time constants are returned; everything else is
    # passed at runtime via sm_configure() after dlopen().
    return {
        "EH_BKT_CAP":  bkt_cap,
        "RSORT_BITS":  bits,
    }


# ---------------------------------------------------------------------------
# Optional path to machine.yaml (set via set_machine_yaml() before first use,
# or via the --machine-yaml CLI flag in the calling script).
# None → search for machine.yaml next to this file, then current directory.
_MACHINE_YAML_PATH: str | None = None

def set_machine_yaml(path: str | None) -> None:
    """Override the machine.yaml path used by _derive_build_constants.

    Call this before any DP function if you want to use a non-default
    machine configuration file (e.g. machine_m3pro.yaml vs machine_m3.yaml).
    Clears the compiled-library cache so the next call recompiles with the
    new constants.

    Parameters
    ----------
    path : str or None
        Absolute or relative path to a YAML file produced by tune_params.py.
        None resets to the default search (machine.yaml next to this file,
        then current directory).
    """
    global _MACHINE_YAML_PATH, _LIB_CACHE
    _MACHINE_YAML_PATH = path
    _LIB_CACHE = {}   # force recompile with new constants

_LIB_CACHE: dict = {}

def _get_lib():
    # Detect cache sizes once; derive -D flags; fold both into the cache key
    # so that recompilation happens automatically on a different machine.
    l1d, l2, l3, cl = _detect_cache_sizes()
    consts = _derive_build_constants(l1d, l2, l3, cl)

    src_hash = hashlib.md5(C_SOURCE.encode()).hexdigest()[:12]
    # Cache key: C source hash + only the two compile-time params.
    # Runtime params (SM_WORKER_CAP, SM_BB_BUF, etc.) no longer affect the binary;
    # changing machine.yaml does not trigger recompilation.
    const_sig = f"EH{consts['EH_BKT_CAP']}_RB{consts['RSORT_BITS']}"
    cache_key = f"{src_hash}_{const_sig}"
    if cache_key in _LIB_CACHE:
        return _LIB_CACHE[cache_key]

    build_dir = os.path.join(tempfile.gettempdir(), f"ham_dp_c_{cache_key}")
    os.makedirs(build_dir, exist_ok=True)
    c_path  = os.path.join(build_dir, "ham_dp.c")
    so_path = os.path.join(build_dir, "ham_dp.so")

    if not os.path.exists(so_path):
        # Report which machine.yaml was used (or that defaults were applied)
        # Re-derive just to get _yaml_used; _derive_build_constants already ran above
        import pathlib as _pl2
        _here2 = _pl2.Path(__file__).parent
        _cands2 = []
        if _MACHINE_YAML_PATH is not None:
            _cands2.append(_pl2.Path(_MACHINE_YAML_PATH))
        _cands2.append(_here2 / "machine.yaml")
        _cands2.append(_pl2.Path.cwd() / "machine.yaml")
        _yaml_report = next((str(p) for p in _cands2 if p.exists()), None)
        if _yaml_report:
            print(f"# machine.yaml: {_yaml_report}", flush=True)
        else:
            print("# machine.yaml: not found — using analytical defaults", flush=True)
        with open(c_path, "w") as f:
            f.write(C_SOURCE)
        d_flags = [f"-D{k}={v}" for k, v in consts.items()]
        import shutil, platform
        compiler = "gcc-15" if shutil.which("gcc-15") else "gcc"
        # Linux needs -lrt for shm_open; macOS has it in libc
        rt_flag = [] if platform.system() == "Darwin" else ["-lrt"]
        result = subprocess.run(
            [compiler, "-O3", "-march=native", "-shared", "-fPIC", "-std=c11",
             "-D_POSIX_C_SOURCE=200809L"]
            + d_flags
            + ["-o", so_path, c_path, "-lpthread"] + rt_flag,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gcc failed:\n{result.stderr.decode()}")

    ffi = cffi.FFI()
    ffi.cdef("""
        void sm_configure(int nthreads, size_t worker_cap, int rbuf_size,
                          int bb_buf, size_t ext_stream_buf, size_t slc_bytes,
                          size_t par_merge_thresh, int steal_factor,
                          size_t rsort_thresh);

        void count_ham_paths_c(
            int n, const int *order, const int *pos, const int *last_s,
            const int *adj_off, const int *adj_dat,
            uint64_t *res_lo, uint64_t *res_hi,
            int verbose,
            const char *checkpoint_path, double checkpoint_secs,
            int step_limit,
            uint64_t *cyc_lo, uint64_t *cyc_hi);

        void count_ham_paths_peh(
            int n, const int *order, const int *pos, const int *last_s,
            const int *adj_off, const int *adj_dat,
            uint64_t *res_lo, uint64_t *res_hi,
            int verbose,
            const char *checkpoint_path, double checkpoint_secs,
            int step_limit,
            uint64_t *cyc_lo, uint64_t *cyc_hi);

        void count_ham_paths_sm(
            int n, const int *order, const int *pos, const int *last_s,
            const int *adj_off, const int *adj_dat,
            uint64_t *res_lo, uint64_t *res_hi,
            int verbose, uint64_t ram_bytes, int instrument,
            uint64_t *cyc_lo, uint64_t *cyc_hi);
    """)
    lib = ffi.dlopen(so_path)

    # Read runtime parameters from machine.yaml (same candidates as _derive_build_constants)
    import yaml as _yaml2, pathlib as _pl2
    _here2 = _pl2.Path(__file__).parent
    _cands2 = []
    if _MACHINE_YAML_PATH is not None:
        _cands2.append(_pl2.Path(_MACHINE_YAML_PATH))
    _cands2.append(_here2 / "machine.yaml")
    _cands2.append(_pl2.Path.cwd() / "machine.yaml")
    _rt = {}
    for _p2 in _cands2:
        if _p2.exists():
            try:
                with open(_p2) as _f2: _rt = _yaml2.safe_load(_f2) or {}
                if _rt: break
            except Exception: pass

    def _rv(key, default): return int(_rt.get(key, default))

    # Detect hardware defaults for fallback (same logic as _derive_build_constants)
    _ram2, _nt2 = _detect_ram_and_threads()
    _l1_2, _l2_2, _l3_2, _ = _detect_cache_sizes()
    _nthreads_def  = _nt2
    _wcap_def      = max(8, min(128, ((_ram2-(4<<30)-(_ram2>>1))//(_nt2*2*24))//(1<<20))) * (1<<20)
    _rbuf_def      = min(128, max(1, 1 << (max(0, (_l2_2//(256*24)).bit_length()-1))))
    _bb_def        = min(4096, max(16, 1 << (max(0, (_l1_2//(_nt2*24)).bit_length()-1))))
    _ext_def       = max(256, _l3_2//(100*24))

    lib.sm_configure(
        _rv("SM_NTHREADS",         _nthreads_def),
        _rv("SM_WORKER_CAP",       _wcap_def),
        _rv("SM_RBUF_SIZE",        _rbuf_def),
        _rv("SM_BB_BUF",           _bb_def),
        _rv("SM_EXT_STREAM_BUF",   _ext_def),
        _rv("SM_SLC_BYTES",        _l3_2),
        _rv("SM_PAR_MERGE_THRESH", 50_000_000),
        _rv("SM_STEAL_FACTOR",     32),
        _rv("RSORT_THRESH",        1 << 19),
    )

    _LIB_CACHE[cache_key] = (lib, ffi)
    return lib, ffi


def count_hamiltonian_paths_c(n: int, order: list, adj: dict,
                               verbose: bool = False,
                               mem_reserve_gb: float = 2.0,
                               load_factor: int = 75,
                               checkpoint_path: str = "",
                               checkpoint_secs: float = 300.0,
                               count_cycles: bool = False):
    """Count undirected Hamiltonian paths (and optionally cycles) via EH frontier DP.

    Parameters
    ----------
    n                : graph size
    order            : vertex ordering (1-indexed), length n.
    adj              : adjacency dict {v: iterable_of_neighbours} (1-indexed).
    verbose          : if True, print per-step profiling to stderr.
    checkpoint_path  : path for checkpoint file ('' = disabled).
    checkpoint_secs  : checkpoint interval in seconds (0 = disable).
    count_cycles     : if True, also count Hamiltonian cycles and return
                       (paths, cycles); otherwise return paths (int).
    """
    import sys
    if verbose and (load_factor != 75 or mem_reserve_gb != 2.0):
        print("# Note: EH backend ignores --load-factor and --mem-reserve "
              "(memory is proportional to actual entry count)", file=sys.stderr)

    lib, ffi = _get_lib()

    pos = [0] * (n + 1)
    for i, v in enumerate(order): pos[v] = i

    last_s = [0] * (n + 1)
    for v in range(1, n + 1):
        nbr_pos = [pos[w] for w in adj[v]]
        last_s[v] = max(nbr_pos) if nbr_pos else pos[v]

    adj_off_list = [0] * (n + 2)
    for v in range(1, n + 1):
        adj_off_list[v + 1] = adj_off_list[v] + len(adj[v])
    adj_dat_list: list = []
    for v in range(1, n + 1):
        adj_dat_list.extend(sorted(adj[v]))

    c_order   = ffi.new("int[]", list(order))
    c_pos     = ffi.new("int[]", pos)
    c_last_s  = ffi.new("int[]", last_s)
    c_adj_off = ffi.new("int[]", adj_off_list)
    c_adj_dat = ffi.new("int[]", adj_dat_list if adj_dat_list else [0])
    c_res_lo  = ffi.new("uint64_t*")
    c_res_hi  = ffi.new("uint64_t*")
    c_ckpt    = ffi.new("char[]", checkpoint_path.encode() if checkpoint_path else b"")

    c_cyc_lo = ffi.new("uint64_t*") if count_cycles else ffi.NULL
    c_cyc_hi = ffi.new("uint64_t*") if count_cycles else ffi.NULL

    lib.count_ham_paths_c(
        n, c_order, c_pos, c_last_s, c_adj_off, c_adj_dat,
        c_res_lo, c_res_hi, int(verbose),
        c_ckpt, ffi.cast("double", checkpoint_secs),
        ffi.cast("int", -1),
        c_cyc_lo, c_cyc_hi,
    )
    lo, hi = int(c_res_lo[0]), int(c_res_hi[0])
    if lo == hi == 0xFFFFFFFFFFFFFFFF:
        raise RuntimeError(
            "Frontier size exceeded MAX_FS_FAST=15 (pathwidth >= 16). "
            "The packed uint64 state encoding is limited to 15 frontier slots."
        )
    paths = (hi << 64) | lo
    if count_cycles:
        cyc_lo2, cyc_hi2 = int(c_cyc_lo[0]), int(c_cyc_hi[0])
        return paths, (cyc_hi2 << 64) | cyc_lo2
    return paths


def count_hamiltonian_paths_peh(n: int, order: list, adj: dict,
                                 verbose: bool = False,
                                 checkpoint_path: str = "",
                                 checkpoint_secs: float = 300.0,
                                 count_cycles: bool = False,
                                 **kwargs):
    """Count Hamiltonian paths via parallel EH (PEH) backend.

    Like the EH backend but with P inserter threads (default 6), one per
    hash-partition of the output state table.  The rsort buffer routes each
    slot's batch to the owning thread's private EHT; no merge is needed at
    the end since each EHT is already fully deduplicated.

    Speedup over single-threaded EH comes from parallelising the bottleneck
    insertion phase.  Memory usage is the same as EH (proportional to the
    actual state count).

    Parameters
    ----------
    n               : graph size
    order           : vertex ordering (1-indexed), length n
    adj             : adjacency dict {v: iterable_of_neighbours}
    verbose         : print per-step profiling to stderr
    checkpoint_path : path for checkpoint file ('' = disabled)
    checkpoint_secs : checkpoint interval in seconds
    """
    lib, ffi = _get_lib()

    pos = [0] * (n + 1)
    for i, v in enumerate(order): pos[v] = i

    last_s = [0] * (n + 1)
    for v in range(1, n + 1):
        nbr_pos = [pos[w] for w in adj[v]]
        last_s[v] = max(nbr_pos) if nbr_pos else pos[v]

    adj_off_list = [0] * (n + 2)
    for v in range(1, n + 1):
        adj_off_list[v + 1] = adj_off_list[v] + len(adj[v])
    adj_dat_list: list = []
    for v in range(1, n + 1):
        adj_dat_list.extend(sorted(adj[v]))

    c_order   = ffi.new("int[]", list(order))
    c_pos     = ffi.new("int[]", pos)
    c_last_s  = ffi.new("int[]", last_s)
    c_adj_off = ffi.new("int[]", adj_off_list)
    c_adj_dat = ffi.new("int[]", adj_dat_list if adj_dat_list else [0])
    c_res_lo  = ffi.new("uint64_t*")
    c_res_hi  = ffi.new("uint64_t*")
    c_ckpt    = ffi.new("char[]", checkpoint_path.encode() if checkpoint_path else b"")

    c_cyc_lo = ffi.new("uint64_t*") if count_cycles else ffi.NULL
    c_cyc_hi = ffi.new("uint64_t*") if count_cycles else ffi.NULL

    lib.count_ham_paths_peh(
        n, c_order, c_pos, c_last_s, c_adj_off, c_adj_dat,
        c_res_lo, c_res_hi, int(verbose),
        c_ckpt, ffi.cast("double", checkpoint_secs),
        ffi.cast("int", -1),
        c_cyc_lo, c_cyc_hi,
    )
    lo, hi = int(c_res_lo[0]), int(c_res_hi[0])
    if lo == hi == 0xFFFFFFFFFFFFFFFF:
        raise RuntimeError(
            "Frontier size exceeded MAX_FS_FAST=15 (pathwidth >= 16)."
        )
    paths = (hi << 64) | lo
    if count_cycles:
        cyc_lo2, cyc_hi2 = int(c_cyc_lo[0]), int(c_cyc_hi[0])
        return paths, (cyc_hi2 << 64) | cyc_lo2
    return paths


def count_hamiltonian_paths_sm(n: int, order: list, adj: dict,
                                verbose: bool = False,
                                instrument: bool = False,
                                checkpoint_path: str = "",
                                checkpoint_secs: float = 300.0,
                                mem_reserve_gb: float = 0.0,
                                count_cycles: bool = False,
                                **kwargs):
    """Count undirected Hamiltonian paths in G_n via the sort-merge frontier DP.

    Parameters
    ----------
    n              : graph size
    order          : vertex ordering (1-indexed), length n.
    adj            : adjacency dict {v: iterable_of_neighbours} (1-indexed).
    verbose        : print per-step frontier DP progress to stderr.
    instrument     : print per-step phase timing breakdown to stderr.
    mem_reserve_gb : pretend the machine has this many fewer GB of RAM, causing
                     the backend to spill to disk (global_ext merge) sooner.
                     Useful to avoid OOM at peak steps on large graphs.
                     E.g. mem_reserve_gb=300 on a 768 GB machine plans for 468 GB.
    """
    import sys
    if checkpoint_path:
        print("# Warning: sort-merge backend does not yet support checkpointing; "
              "checkpoint_path ignored.", file=sys.stderr)
    lib, ffi = _get_lib()
    ram, _ = _detect_ram_and_threads()
    if mem_reserve_gb > 0:
        reserve = int(mem_reserve_gb * (1 << 30))
        ram = max(int(4 << 30), ram - reserve)  # never go below 4 GB
        print(f"# mem_reserve_gb={mem_reserve_gb:.1f}: "
              f"effective RAM for spill decisions = {ram / (1<<30):.0f} GB",
              file=sys.stderr)

    pos = [0] * (n + 1)
    for i, v in enumerate(order): pos[v] = i

    last_s = [0] * (n + 1)
    for v in range(1, n + 1):
        nbr_pos = [pos[w] for w in adj[v]]
        last_s[v] = max(nbr_pos) if nbr_pos else pos[v]

    adj_off_list = [0] * (n + 2)
    for v in range(1, n + 1):
        adj_off_list[v + 1] = adj_off_list[v] + len(adj[v])
    adj_dat_list: list = []
    for v in range(1, n + 1):
        adj_dat_list.extend(sorted(adj[v]))

    c_order   = ffi.new("int[]", list(order))
    c_pos     = ffi.new("int[]", pos)
    c_last_s  = ffi.new("int[]", last_s)
    c_adj_off = ffi.new("int[]", adj_off_list)
    c_adj_dat = ffi.new("int[]", adj_dat_list if adj_dat_list else [0])
    c_res_lo  = ffi.new("uint64_t*")
    c_res_hi  = ffi.new("uint64_t*")

    c_cyc_lo = ffi.new("uint64_t*") if count_cycles else ffi.NULL
    c_cyc_hi = ffi.new("uint64_t*") if count_cycles else ffi.NULL

    lib.count_ham_paths_sm(
        n, c_order, c_pos, c_last_s, c_adj_off, c_adj_dat,
        c_res_lo, c_res_hi, int(verbose), ram, int(instrument),
        c_cyc_lo, c_cyc_hi,
    )
    lo, hi = int(c_res_lo[0]), int(c_res_hi[0])
    if lo == hi == 0xFFFFFFFFFFFFFFFF:
        raise RuntimeError(
            "Frontier size exceeded MAX_FS_FAST=15 (pathwidth >= 16)."
        )
    paths = (hi << 64) | lo
    if count_cycles:
        cyc_lo2, cyc_hi2 = int(c_cyc_lo[0]), int(c_cyc_hi[0])
        return paths, (cyc_hi2 << 64) | cyc_lo2
    return paths


def partial_dp_time_c(n: int, order: list, adj: dict,
                      step_limit: int) -> float:
    """Run the frontier DP for at most *step_limit* steps and return elapsed ms.

    Used to rank candidate orderings by early-phase DP cost without paying
    for the full computation.  Typically step_limit = n // 2 captures the
    expensive peak steps while running in a fraction of the full DP time.

    Parameters
    ----------
    n          : graph size
    order      : vertex ordering (1-indexed), length n
    adj        : adjacency dict {v: iterable_of_neighbours}
    step_limit : stop after this many steps (must be >= 1)

    Returns
    -------
    Elapsed wall-clock time in milliseconds for those steps.
    """
    lib, ffi = _get_lib()

    pos = [0] * (n + 1)
    for i, v in enumerate(order): pos[v] = i

    last_s = [0] * (n + 1)
    for v in range(1, n + 1):
        nbr_pos = [pos[w] for w in adj[v]]
        last_s[v] = max(nbr_pos) if nbr_pos else pos[v]

    adj_off_list = [0] * (n + 2)
    for v in range(1, n + 1):
        adj_off_list[v + 1] = adj_off_list[v] + len(adj[v])
    adj_dat_list: list = []
    for v in range(1, n + 1):
        adj_dat_list.extend(sorted(adj[v]))

    c_order   = ffi.new("int[]", list(order))
    c_pos     = ffi.new("int[]", pos)
    c_last_s  = ffi.new("int[]", last_s)
    c_adj_off = ffi.new("int[]", adj_off_list)
    c_adj_dat = ffi.new("int[]", adj_dat_list if adj_dat_list else [0])
    c_res_lo  = ffi.new("uint64_t*")
    c_res_hi  = ffi.new("uint64_t*")
    c_empty   = ffi.new("char[]", b"")

    lib.count_ham_paths_c(
        n, c_order, c_pos, c_last_s, c_adj_off, c_adj_dat,
        c_res_lo, c_res_hi, 0,      # verbose=False
        c_empty, ffi.cast("double", 0.0),
        ffi.cast("int", step_limit),
    )
    # Decode result:
    #   res_hi == UINT64_MAX-1 : partial-run sentinel — res_lo holds elapsed ms
    #   res_hi == UINT64_MAX   : MAX_FS_FAST overflow (ordering exceeds pw limit)
    #   otherwise              : step_limit >= n, full DP ran (shouldn't happen)
    import struct
    hi = int(c_res_hi[0])
    if hi == 0xFFFFFFFFFFFFFFFE:       # UINT64_MAX - 1: partial run sentinel
        elapsed_ms = struct.unpack('d', struct.pack('Q', int(c_res_lo[0])))[0]
    elif hi == 0xFFFFFFFFFFFFFFFF:     # UINT64_MAX: overflow — ordering is unusable
        elapsed_ms = float('inf')
    else:
        # step_limit >= n, so the full DP completed — measure elapsed via res_lo/hi
        # This shouldn't happen in normal use but return inf to signal "use full DP"
        elapsed_ms = float('inf')
    return elapsed_ms


    import sys, time
    start_n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    end_n   = int(sys.argv[2]) if len(sys.argv) > 2 else start_n
    try:
        from .ham_ordering import build_graph, best_bfs_order, frontier_stats
    except ImportError:
        from math import isqrt
        def build_graph(n):
            adj = {v: set() for v in range(1, n+1)}
            for i in range(1, n+1):
                for j in range(i+1, n+1):
                    s=i+j; r=isqrt(s)
                    if r*r==s: adj[i].add(j); adj[j].add(i)
            return adj
        def best_bfs_order(adj, n):
            import networkx as nx
            G = nx.Graph([(u,v) for u,ns in adj.items() for v in ns])
            nodes = list(range(1,n+1))
            best=(999,999,None)
            for s in nodes:
                o=list(nx.bfs_tree(G,s).nodes())+sorted(set(nodes)-set(nx.bfs_tree(G,s).nodes()))
                pos={v:i for i,v in enumerate(o)}
                fw=[sum(1 for u in o[:i+1] if any(pos[nb]>i for nb in adj[u])) for i in range(n)]
                if (max(fw),sum(fw))<best[:2]: best=(max(fw),sum(fw),o)
            return best[2]
        def frontier_stats(adj, order):
            pos={v:i for i,v in enumerate(order)}
            fw=[sum(1 for u in order[:i+1] if any(pos[nb]>i for nb in adj[u])) for i in range(len(order))]
            return max(fw),sum(fw)
    print("# Compiling …", flush=True); _get_lib(); print("# Ready.", flush=True)
    for n in range(start_n, end_n + 1):
        adj=build_graph(n); order=best_bfs_order(adj,n); mx,_=frontier_stats(adj,order)
        t0=time.time(); count=count_hamiltonian_paths_c(n,order,adj,verbose=True)
        print(f"n={n:3d}  max_fw={mx:2d}  ham_paths={count}  ({time.time()-t0:.3f}s)", flush=True)
