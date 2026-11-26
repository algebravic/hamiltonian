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

import os, hashlib, math, subprocess, tempfile, cffi

import pathlib as _pathlib
_HERE = _pathlib.Path(__file__).parent
# C source variants.  "key16" (default): 16 frontier slots, no flag bits
# (ham_dp_c.c).  "legacy15": the original 15-slot key with the NC/MARKER flag
# bits (ham_dp_c_legacy15.c), kept for comparison and as a fallback.
_C_VARIANTS = {"key16": ("ham_dp_c.c", 16), "legacy15": ("ham_dp_c_legacy15.c", 15)}
_C_VARIANT = os.environ.get("HAM_DP_VARIANT", "key16")
if _C_VARIANT not in _C_VARIANTS:
    raise ValueError(f"HAM_DP_VARIANT must be one of {sorted(_C_VARIANTS)}")
C_SOURCE = (_HERE / _C_VARIANTS[_C_VARIANT][0]).read_text()


def set_c_variant(name: str) -> None:
    """Select the C implementation: "key16" (default) or "legacy15"."""
    global _C_VARIANT, C_SOURCE, _LIB_CACHE, MAX_STATE_SLOTS
    if name not in _C_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_C_VARIANTS)}")
    _C_VARIANT = name
    C_SOURCE = (_HERE / _C_VARIANTS[name][0]).read_text()
    MAX_STATE_SLOTS = _C_VARIANTS[name][1]
    _LIB_CACHE = {}

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
        # Apple Silicon: the unqualified keys may describe a different core
        # type; the P-core values (perflevel0) are the ones the DP runs on.
        l1  = sysctl_int("hw.perflevel0.l1dcachesize") or sysctl_int("hw.l1dcachesize")
        l2  = sysctl_int("hw.perflevel0.l2cachesize") or sysctl_int("hw.l2cachesize")
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


def _parse_cpu_list(text: str) -> int:
    """Number of CPUs in a sysfs cpu list such as '0-3,32-35'."""
    n = 0
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            n += int(b) - int(a) + 1
        else:
            n += 1
    return n


def _detect_cache_sharing() -> dict:
    """Return {'l1', 'l2', 'l3': logical CPUs sharing one instance,
    'logical': logical CPUs the DP may use}.

    Linux  — /sys/devices/system/cpu/cpu0/cache/index*/shared_cpu_list
    macOS  — L1 private (no SMT); L2 shared by hw.perflevel0.cpusperl2;
             the SLC is shared by the whole chip.
    Unknown levels default to 1 (private), which reproduces the old
    'each thread owns the cache' sizing.
    """
    import platform
    share = {"l1": 1, "l2": 1, "l3": 1, "logical": os.cpu_count() or 1}
    if platform.system() == "Linux":
        base = "/sys/devices/system/cpu/cpu0/cache"
        for idx in range(10):
            d = f"{base}/index{idx}"
            if not os.path.isdir(d):
                break
            try:
                level = int(open(f"{d}/level").read())
                ctype = open(f"{d}/type").read().strip()
                if level == 1 and "Instruction" in ctype:
                    continue
                share[f"l{level}"] = _parse_cpu_list(open(f"{d}/shared_cpu_list").read())
            except Exception:
                continue
    elif platform.system() == "Darwin":
        def sysctl_int(key):
            r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            try: return int(r.stdout.strip()) if r.returncode == 0 else None
            except ValueError: return None
        p = sysctl_int("hw.perflevel0.logicalcpu") or sysctl_int("hw.logicalcpu") or 1
        share["logical"] = p
        share["l2"] = sysctl_int("hw.perflevel0.cpusperl2") or p
        share["l3"] = sysctl_int("hw.logicalcpu") or p
    return share


def _per_thread_share(size: int, sharers: int, nthreads: int, logical: int) -> int:
    """Bytes of one cache instance available to one DP thread.

    With nthreads spread evenly over `logical` CPUs, an instance shared by
    `sharers` CPUs hosts about ceil(nthreads * sharers / logical) of our
    threads (at least 1, at most `sharers`).
    """
    if sharers <= 1 or logical <= 0:
        return size
    k = -(-nthreads * sharers // logical)
    k = max(1, min(sharers, k))
    return size // k


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
            curr_tab + SM_NTHREADS × 2 × SM_WORKER_CAP × 16B + headroom
        Solve for cap given available RAM, leaving OS_HEADROOM_GB free:
            cap = (ram - os_headroom - curr_max) / (nthreads × 2 × 16)
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
    SM_ENTRY     = 16          # bytes per SMEntry {u64 key; u64 val}
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
    curr_max_bytes = 553_000_000 * SM_ENTRY   # 8.8 GB

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
    # SM_RBUF_SIZE: fill L2 with RadixBuf (256 buckets × size × 16B ≤ L2)
    rbuf_default  = 1
    while (rbuf_default * 2) * 256 * SM_ENTRY <= l2: rbuf_default *= 2
    rbuf_default  = min(rbuf_default, 128)   # practical cap

    # SM_BB_BUF: fill L1 with block-merge buffers (P × size × 16B ≤ L1)
    bb_default    = 1
    while (bb_default * 2) * n_threads * SM_ENTRY <= l1d: bb_default *= 2
    bb_default    = min(bb_default, 512)

    # SM_EXT_STREAM_BUF: floor; dynamic formula in C overrides per step
    ext_default   = max(256, l3 // (100 * SM_ENTRY))   # rough: 1% of SLC each

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

# ---------------------------------------------------------------------------
# Build cache.  The compiled library is keyed by a SHA-256 over everything
# that determines the binary: the C source, the variant, the compiler's
# resolved path and `--version`, every flag and -D constant, and the
# platform.  Builds go to a persistent cache directory, under a file lock,
# and are renamed into place only when complete.
#   cache dir : $HAM_DP_CACHE, else $XDG_CACHE_HOME/hamiltonian/ham_dp_c,
#               else ~/.cache/hamiltonian/ham_dp_c
#   compiler  : set_compiler(path) / $HAM_DP_CC, else gcc-15 if on PATH,
#               else gcc
#   CLI       : python -m hamiltonian.claude.ham_dp_c --list-cache
#               python -m hamiltonian.claude.ham_dp_c --clear-cache
# ---------------------------------------------------------------------------
_CC_OVERRIDE: str | None = None
_FORCE_REBUILD = False
_CC_IDENT: dict = {}


def set_compiler(path: str | None) -> None:
    """Use this C compiler (None: $HAM_DP_CC, else gcc-15, else gcc)."""
    global _CC_OVERRIDE, _LIB_CACHE
    _CC_OVERRIDE = path
    _LIB_CACHE = {}


def force_rebuild(flag: bool = True) -> None:
    """Recompile on the next use even if a cached build exists."""
    global _FORCE_REBUILD, _LIB_CACHE
    _FORCE_REBUILD = flag
    _LIB_CACHE = {}


def cache_dir() -> str:
    d = os.environ.get("HAM_DP_CACHE")
    if not d:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        d = os.path.join(base, "hamiltonian", "ham_dp_c")
    os.makedirs(d, exist_ok=True)
    return d


def _compiler() -> str:
    import shutil
    cc = _CC_OVERRIDE or os.environ.get("HAM_DP_CC")
    if not cc:
        cc = "gcc-15" if shutil.which("gcc-15") else "gcc"
    return cc


def _compiler_ident(cc: str) -> tuple:
    """(resolved path, first line of --version, full --version text)."""
    if cc not in _CC_IDENT:
        import shutil
        path = shutil.which(cc) or cc
        real = os.path.realpath(path)
        try:
            v = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
        except OSError:
            v = ""
        _CC_IDENT[cc] = (real, (v.splitlines() or ["?"])[0].strip(), v)
    return _CC_IDENT[cc]


def _build_flags(consts: dict) -> list:
    import platform
    rt_flag = [] if platform.system() == "Darwin" else ["-lrt"]
    return (["-O3", "-march=native", "-shared", "-fPIC", "-std=c11",
             "-D_POSIX_C_SOURCE=200809L"]
            + [f"-D{k}={v}" for k, v in sorted(consts.items())]
            + ["-lpthread"] + rt_flag)


def _build_key(cc: str, flags: list) -> tuple:
    import json, platform
    real, first, full = _compiler_ident(cc)
    inputs = {
        "source_sha256": hashlib.sha256(C_SOURCE.encode()).hexdigest(),
        "variant": _C_VARIANT,
        "compiler": real, "compiler_version": full,
        "flags": flags,
        "system": platform.system(), "machine": platform.machine(),
        "release": platform.release() if platform.system() == "Darwin" else "",
    }
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    meta = dict(inputs, compiler_version=first, cc=cc)
    return key, meta


def list_cache() -> list:
    """[(key, meta dict, size bytes, mtime)] of the cached builds."""
    import json
    out = []
    d = cache_dir()
    for k in sorted(os.listdir(d)):
        sub = os.path.join(d, k)
        so = os.path.join(sub, "ham_dp.so")
        if not os.path.isfile(so):
            continue
        try:
            meta = json.load(open(os.path.join(sub, "meta.json")))
        except Exception:
            meta = {}
        st = os.stat(so)
        out.append((k, meta, st.st_size, st.st_mtime))
    return out


def clear_cache() -> int:
    """Delete every cached build; returns the number removed."""
    import shutil
    d = cache_dir(); n = 0
    for k in os.listdir(d):
        p = os.path.join(d, k)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True); n += 1
    # Builds of the old scheme (before 2026-11-26): $TMPDIR/ham_dp_c_<md5>_<consts>
    import glob
    for p in glob.glob(os.path.join(tempfile.gettempdir(), "ham_dp_c_*")):
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True); n += 1
    global _LIB_CACHE
    _LIB_CACHE = {}
    return n


def _get_lib():
    global _FORCE_REBUILD
    # Detect cache sizes once; derive -D flags; fold both into the cache key
    # so that recompilation happens automatically on a different machine.
    l1d, l2, l3, cl = _detect_cache_sizes()
    consts = _derive_build_constants(l1d, l2, l3, cl)

    cc = _compiler()
    flags = _build_flags(consts)
    cache_key, meta = _build_key(cc, flags)
    if cache_key in _LIB_CACHE and not _FORCE_REBUILD:
        return _LIB_CACHE[cache_key]

    build_dir = os.path.join(cache_dir(), cache_key)
    os.makedirs(build_dir, exist_ok=True)
    c_path  = os.path.join(build_dir, "ham_dp.c")
    so_path = os.path.join(build_dir, "ham_dp.so")

    # Report which machine.yaml was used (or that defaults were applied).
    import pathlib as _pl2
    _here2 = _pl2.Path(__file__).parent
    _cands2 = []
    if _MACHINE_YAML_PATH is not None:
        _cands2.append(_pl2.Path(_MACHINE_YAML_PATH))
    _cands2.append(_here2 / "machine.yaml")
    _cands2.append(_pl2.Path.cwd() / "machine.yaml")
    _yaml_report = next((str(p) for p in _cands2 if p.exists()), None)
    print(f"# machine.yaml: {_yaml_report or 'not found — using analytical defaults'}",
          flush=True)

    import fcntl, json
    with open(os.path.join(build_dir, "build.lock"), "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)          # one builder per key
        status = "cached"
        if _FORCE_REBUILD or not os.path.exists(so_path):
            tmp_c = os.path.join(build_dir, f"ham_dp.{os.getpid()}.tmp.c")
            tmp_so = os.path.join(build_dir, f"ham_dp.{os.getpid()}.tmp.so")
            with open(tmp_c, "w") as f:
                f.write(C_SOURCE)
            # flags = compile flags + link libs; put -o and the source before libs
            libs = [f for f in flags if f.startswith("-l")]
            cflags = [f for f in flags if not f.startswith("-l")]
            cmd = [cc] + cflags + ["-o", tmp_so, tmp_c] + libs
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                for t in (tmp_c, tmp_so):
                    try: os.unlink(t)
                    except OSError: pass
                raise RuntimeError(f"{cc} failed:\n{result.stderr.decode()}")
            os.replace(tmp_c, c_path)
            os.replace(tmp_so, so_path)               # atomic: never half-built
            with open(os.path.join(build_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=1, sort_keys=True)
            status = "compiled"
        fcntl.flock(lockf, fcntl.LOCK_UN)
    _FORCE_REBUILD = False                        # a forced rebuild happens once
    print(f"# ham_dp_c: {status} variant={_C_VARIANT} key={cache_key} "
          f"cc={meta['compiler']} ({meta['compiler_version']}) "
          f"flags={' '.join(f for f in flags if not f.startswith('-l'))}", flush=True)

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

        int  sm_overflow_detected(void);
        int  sm_overflow_step(void);
        void sm_set_modulus(uint64_t p);
        void sm_set_word_bits(int b);
        void sm_set_spill_dir(const char *dir);
        void sm_set_crt(const uint64_t *moduli, int k, uint64_t snap_bytes);
        int  sm_crt_replay_from(void);
        uint64_t sm_crt_residue(int i, int which);

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
    _nthreads_run  = _rv("SM_NTHREADS", _nthreads_def)
    # Per-thread cache shares: SMT siblings share L1/L2 on x86, a cluster
    # shares L2 on Apple Silicon, a CCX/chip shares L3/SLC.
    _sh = _detect_cache_sharing()
    _l1_t = _per_thread_share(_l1_2, _sh["l1"], _nthreads_run, _sh["logical"])
    _l2_t = _per_thread_share(_l2_2, _sh["l2"], _nthreads_run, _sh["logical"])
    _E = 16                                  # sizeof(SMEntry) = {u64 key; u64 val}
    _wcap_def      = max(8, min(128, ((_ram2-(4<<30)-(_ram2>>1))//(_nt2*2*_E))//(1<<20))) * (1<<20)
    # 256 radix write buffers per thread within the thread's L2 share.
    _rbuf_def      = min(128, max(1, 1 << (max(0, (_l2_t//(256*_E)).bit_length()-1))))
    # K = P streams per merge thread, block buffers within its L1 share.
    _bb_def        = min(4096, max(16, 1 << (max(0, (_l1_t//(_nt2*_E)).bit_length()-1))))
    # SM_SLC_BYTES / SM_EXT_STREAM_BUF size the ext-merge read-ahead (I/O),
    # not a cache budget; they keep the whole-LLC default.
    _ext_def       = max(256, _l3_2//(100*_E))

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
    print(f"# ham_dp_c: SM_NTHREADS={_rv('SM_NTHREADS', _nthreads_def)} "
          f"SM_WORKER_CAP={_rv('SM_WORKER_CAP', _wcap_def)} "
          f"SM_RBUF_SIZE={_rv('SM_RBUF_SIZE', _rbuf_def)} "
          f"SM_BB_BUF={_rv('SM_BB_BUF', _bb_def)} "
          f"SM_PAR_MERGE_THRESH={_rv('SM_PAR_MERGE_THRESH', 50_000_000)}", flush=True)

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

    _check_width(order, adj)
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
            f"Frontier exceeded the {MAX_STATE_SLOTS} slots of the 64-bit state key. "
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
    _check_width(order, adj)
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
            f"Frontier exceeded the {MAX_STATE_SLOTS} slots of the 64-bit state key."
        )
    paths = (hi << 64) | lo
    if count_cycles:
        cyc_lo2, cyc_hi2 = int(c_cyc_lo[0]), int(c_cyc_hi[0])
        return paths, (cyc_hi2 << 64) | cyc_lo2
    return paths


# ---------------------------------------------------------------------------
# 64-bit state counts: residues and the Chinese remainder theorem
# ---------------------------------------------------------------------------
# The sort-merge DP stores each state's count in a u64.  A state's count is
# the number of partial path systems with that frontier state and is not
# bounded by H(n), so it can wrap even when H(n) < 2^64 (n = 74, kpw
# ordering).  The DP only adds counts, so with wrapping arithmetic the result
# is exact mod 2^64; replays mod primes p_i < 2^62 give it mod p_i, and the
# CRT recovers H(n) exactly provided H(n) < 2^64 * prod p_i.  That
# hypothesis is certified by the Bregman-Minc bound (hamiltonian/bounds/
# bregman.py): adding a vertex joined to everything turns each directed
# Hamiltonian path into a cycle cover, so
#     2 H <= per(A') <= prod_v ((d_v + 1)!)^(1/(d_v + 1)) * (n!)^(1/n),
# and the number of Hamiltonian cycles is at most H / n <= H.

MAX_STATE_SLOTS = _C_VARIANTS[_C_VARIANT][1]   # MAX_FS_FAST of the selected C variant


def in_step_width(order: list, adj: dict) -> int:
    """Largest frontier during a step: the vertices still waiting for a later
    neighbour plus the vertex being introduced.  This is what must fit in the
    state key; it is one more than the vertex separation of the ordering
    (max_fw / pw as reported by ham_count) whenever the maximum is followed
    by an introduction."""
    pos = {v: i for i, v in enumerate(order)}
    last = {v: max([pos[w] for w in adj[v]] + [pos[v]]) for v in order}
    live, best = 0, 0
    ends = {}
    for i, v in enumerate(order):
        live += 1
        best = max(best, live)
        ends[last[v]] = ends.get(last[v], 0) + 1
        live -= ends.pop(i, 0)
    return best


def _check_width(order: list, adj: dict) -> None:
    w = in_step_width(order, adj)
    if w > MAX_STATE_SLOTS:
        raise RuntimeError(
            f"This ordering needs {w} frontier slots during a step (vertex "
            f"separation {w - 1}); the 64-bit state key of variant "
            f"{_C_VARIANT!r} holds {MAX_STATE_SLOTS}. "
            f"Use an ordering of vertex separation <= {MAX_STATE_SLOTS - 1}.")


class SMOverflow(OverflowError):
    """State counts wrapped mod 2^64 and the CRT was not requested.

    residue        : number of paths mod 2^64 (2^_word_bits in tests)
    cycles_residue : number of cycles mod 2^64 (None unless counted)
    step           : first step at which a count wrapped
    """
    def __init__(self, msg, residue, cycles_residue, step):
        super().__init__(msg)
        self.residue, self.cycles_residue, self.step = residue, cycles_residue, step


def _is_prime(m: int) -> bool:
    """Deterministic Miller-Rabin for m < 3.3e24 (first 12 prime bases)."""
    if m < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for q in small:
        if m % q == 0:
            return m == q
    d, r = m - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    for a in small:
        x = pow(a, d, m)
        if x in (1, m - 1):
            continue
        for _ in range(r - 1):
            x = x * x % m
            if x == m - 1:
                break
        else:
            return False
    return True


_CRT_PRIMES: list = []

def crt_primes(k: int) -> list:
    """The k largest primes <= 2^61 - 1 (a Mersenne prime first)."""
    if len(_CRT_PRIMES) < k:
        c = (1 << 61) - 1 if not _CRT_PRIMES else _CRT_PRIMES[-1] - 2
        while len(_CRT_PRIMES) < k:
            if _is_prime(c):
                _CRT_PRIMES.append(c)
            c -= 2
    return _CRT_PRIMES[:k]


def bregman_log2_bound(n: int, adj: dict) -> float:
    """log2 of the Bregman-Minc upper bound for the number of (undirected)
    Hamiltonian paths of the graph adj on vertices 1..n."""
    from math import lgamma, log
    s = sum(lgamma(len(adj[v]) + 2) / (len(adj[v]) + 1) for v in range(1, n + 1))
    s += lgamma(n + 1) / n - log(2)
    return s / log(2)


def crt_combine(residues: list, moduli: list) -> int:
    """The unique 0 <= x < prod(moduli) with x = r_i mod m_i (Garner)."""
    x, m = 0, 1
    for r, q in zip(residues, moduli):
        t = ((r - x) * pow(m, -1, q)) % q
        x += m * t
        m *= q
    return x


def count_hamiltonian_paths_sm(n: int, order: list, adj: dict,
                                verbose: bool = False,
                                instrument: bool = False,
                                checkpoint_path: str = "",
                                checkpoint_secs: float = 300.0,
                                mem_reserve_gb: float = 0.0,
                                count_cycles: bool = False,
                                ram_bytes: int | None = None,
                                on_overflow: str = "crt",
                                crt_snap_bytes: int | None = None,
                                spill_dir: str | None = None,
                                _word_bits: int = 64,
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
    ram_bytes      : use this instead of the detected physical RAM for the
                     RAM/ext and merge-budget decisions (tests, what-if runs).
                     mem_reserve_gb is applied on top of it.
    on_overflow    : what to do if a state count wraps mod 2^64.
                     "crt"     (default) recover the exact value: replay the
                               steps from the last snapshot before the first
                               wrap mod enough primes p_i < 2^62 that
                               2^64 * prod p_i exceeds the Bregman bound, or,
                               if no snapshot was kept, rerun the whole DP
                               mod each p_i; combine by the CRT.
                     "residue" raise SMOverflow carrying the result mod 2^64
                               and the first wrapping step.
    crt_snap_bytes : most RAM the CRT anchor snapshot may use; a larger
                     table is written once to spill_dir instead (default
                     min(RAM/16, 64 GiB)).  Rolling snapshots later in the
                     run use at most min(1 GiB, crt_snap_bytes).
    spill_dir      : directory for runs that do not fit in RAM and for disk
                     snapshots (default $TMPDIR, else /tmp).  The files are
                     unlinked at creation, so nothing is left behind.
    _word_bits     : test hook only: emulate 2^b-bit counts (b < 64) so that
                     wraps and the CRT path can be exercised at small n.
    """
    import sys
    if checkpoint_path:
        print("# Warning: sort-merge backend does not yet support checkpointing; "
              "checkpoint_path ignored.", file=sys.stderr)
    _check_width(order, adj)
    lib, ffi = _get_lib()
    ram, _ = _detect_ram_and_threads()
    if ram_bytes:
        ram = int(ram_bytes)
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

    if on_overflow not in ("crt", "residue"):
        raise ValueError(f"on_overflow must be 'crt' or 'residue', not {on_overflow!r}")
    bound2 = bregman_log2_bound(n, adj)
    k = 0
    W = _word_bits if 8 <= _word_bits < 64 else 64
    while W + sum(log2_p for log2_p in
                   (math.log2(q) for q in crt_primes(k))) < bound2 + 1.0:
        k += 1
    primes = crt_primes(k)
    snap = crt_snap_bytes if crt_snap_bytes is not None else min(64 << 30, ram // 16)
    lib.sm_set_spill_dir((spill_dir or "").encode())

    def _run(modulus=0, crt=()):
        lib.sm_set_modulus(modulus)
        c_mod = ffi.new("uint64_t[]", list(crt) or [0])
        lib.sm_set_crt(c_mod, len(crt), snap if crt else 0)
        try:
            lib.count_ham_paths_sm(
                n, c_order, c_pos, c_last_s, c_adj_off, c_adj_dat,
                c_res_lo, c_res_hi, int(verbose), ram, int(instrument),
                c_cyc_lo, c_cyc_hi,
            )
        finally:
            lib.sm_set_modulus(0)
            lib.sm_set_crt(c_mod, 0, 0)
        lo_, hi_ = int(c_res_lo[0]), int(c_res_hi[0])
        if lo_ == hi_ == 0xFFFFFFFFFFFFFFFF:
            raise RuntimeError(f"Frontier exceeded the {MAX_STATE_SLOTS} slots of the 64-bit state key.")
        cyc_ = ((int(c_cyc_hi[0]) << 64) | int(c_cyc_lo[0])) if count_cycles else None
        return (hi_ << 64) | lo_, cyc_

    lib.sm_set_word_bits(W)
    try:
        paths, cycles = _run(crt=primes if on_overflow == "crt" else ())
    except BaseException:
        lib.sm_set_word_bits(64); raise
    if lib.sm_overflow_detected():
        step = lib.sm_overflow_step()
        r_paths = paths % (1 << W)
        r_cyc = cycles % (1 << W) if count_cycles else None
        if on_overflow == "residue":
            lib.sm_set_word_bits(64)
            raise SMOverflow(
                f"Sort-merge DP: a state count wrapped mod 2^{W} (first at step {step}). "
                f"Paths mod 2^{W} = {r_paths}"
                + (f", cycles mod 2^{W} = {r_cyc}" if count_cycles else "")
                + ". Use on_overflow='crt' for the exact value.",
                r_paths, r_cyc, step)
        frm = lib.sm_crt_replay_from()
        if frm >= 0:
            res_p = [int(lib.sm_crt_residue(i, 0)) for i in range(k)]
            res_c = [int(lib.sm_crt_residue(i, 1)) for i in range(k)]
            how = f"replayed steps {frm}..{n-1}"
        else:
            res_p, res_c = [], []
            lib.sm_set_word_bits(64)          # mod-p runs never wrap
            for q in primes:
                pp, cc = _run(modulus=q)
                res_p.append(pp % q)
                res_c.append(cc % q if count_cycles else 0)
            how = "full reruns (no snapshot before the wrap)"
        mods = [1 << W] + primes
        paths = crt_combine([r_paths] + res_p, mods)
        if count_cycles:
            cycles = crt_combine([r_cyc] + res_c, mods)
        M2 = W + sum(math.log2(q) for q in primes)
        # Certification: the true value is < 2^bound2 < prod(mods); the CRT
        # value is the unique residue below prod(mods), so it is the count.
        assert paths < 2.0 ** (bound2 + 1.0), "CRT result exceeds the Bregman bound"
        print(f"# sort-merge: counts wrapped mod 2^{W} (first at step {step}); "
              f"exact value by CRT with {k} prime(s) < 2^62 ({how}); "
              f"Bregman bound 2^{bound2:.1f} < modulus 2^{M2:.1f}", file=sys.stderr)
    lib.sm_set_word_bits(64)
    if count_cycles:
        return paths, cycles
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
        ffi.NULL, ffi.NULL,         # no cycle counting
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


def _cache_main(argv=None) -> int:
    import argparse, time
    ap = argparse.ArgumentParser(prog="python -m hamiltonian.claude.ham_dp_c",
                                 description="Manage the compiled ham_dp_c libraries.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list-cache", action="store_true", help="list cached builds")
    g.add_argument("--clear-cache", action="store_true", help="delete all cached builds")
    g.add_argument("--build", action="store_true",
                   help="build (or report the cached build of) the current configuration")
    ap.add_argument("--cc", default=None, help="compiler for --build")
    ap.add_argument("--variant", default=None, choices=sorted(_C_VARIANTS))
    a = ap.parse_args(argv)
    if a.list_cache:
        print(f"cache directory: {cache_dir()}")
        for k, meta, size, mt in list_cache():
            print(f"{k}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(mt))}  "
                  f"{size/1e3:7.0f} kB  variant={meta.get('variant','?')}  "
                  f"cc={meta.get('compiler','?')} ({meta.get('compiler_version','?')})")
    elif a.clear_cache:
        print(f"removed {clear_cache()} build(s) from {cache_dir()}")
    else:
        if a.cc: set_compiler(a.cc)
        if a.variant: set_c_variant(a.variant)
        _get_lib()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cache_main())
