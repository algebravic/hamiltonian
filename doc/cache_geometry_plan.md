# Cache geometry (line size, associativity, sharing) in `ham_dp_c` — analysis and plan

Status (2026-11-25): **A0, A1, B1, B2, D and `src/C/cacheprobe.c` are implemented**; see §6 for what was done, the deviations from this plan and the test results. B3, C, E, F and G are not implemented.

## 0. Short answer

Partly yes, but less than one might hope, and not where the time is going at n = 75.

* **Line size** matters, in three places:
  * padding of per-thread structures (false sharing);
  * the granularity of the radix-sort write buffers;
  * avoiding read-for-ownership on stores that overwrite whole lines.

  Two of these are already half-handled. The `SM_CACHE_ALIGN` attribute exists, but it is defeated by `calloc`; see B1.
* **Associativity** matters essentially only for **L1**. L1 is virtually indexed (VIPT), with the set index inside the page offset: on x86, 64 sets × 64 B = 4 KiB = the page; on Apple, 16 KiB pages cover the L1 index. So power-of-two strides in *virtual* addresses alias deterministically in L1.

  For L2/L3/SLC, the set-index bits above the page offset come from the physical frame number. With 4 KiB (Linux) or 16 KiB (macOS) pages that is effectively random, so virtual layout cannot cause or cure conflicts there. The exception is huge pages (Linux THP, 2 MiB). In addition, Zen 4's L3 and Apple's SLC are hashed or memory-side.

  So an *experimentally measured* L2 associativity would mostly serve as a sanity check, not as a layout parameter. I'll still measure it, since you asked and it is cheap to record, but the plan does not depend on it.
* **Sharing** — which threads share which cache — matters more than associativity for "threads stepping on each other". Examples:
  * Apple: the P-cluster L2 is shared by `hw.perflevel0.cpusperl2` cores.
  * Zen 4: the L3 is shared per CCD. On the EPYC 9124 that is 4 cores = 8 threads; see §0a.
  * SMT siblings share L1/L2.

  The current formulas size several buffers as if each thread owned the whole cache.

**A bigger finding, unrelated to geometry.** In `profile_75log.txt` (Linux, 64 workers), **51.7 % of total DP time (3026 s of 5851 s) is the per-worker `sm_merge_runs` loop, which runs serially on the main thread** (`ham_dp_c.c` ≈ l. 3033, the "RAM path, Stage 1"). The instrumentation already labels it `merge_runs(seq)`, and its `total=` column accounts for the unexplained ~60–70 % of every RAM-mode big step.

For comparison, the ext-mode steps merge all runs in one parallel pass (`global_ext_par`) and show no such gap. Fixing this (item A), with the parallelism chosen to fit the memory budget, should save far more than everything else here combined. It is a parallelisation fix, not a cache fix.

## 0a. The Linux machine (from `data/linux_arch.txt`)

The machine is 2 × AMD EPYC 9124 (Zen 4, Genoa): **16 cores per socket, SMT 2, i.e. 32 physical cores and 64 logical CPUs**, on 2 NUMA nodes. The sysfs cache entries for cpu0 are:

| level | size | line | ways | sets | way size | shared by (cpu0) |
|---|---|---|---|---|---|---|
| L1d | 32 KiB | 64 | 8 | 64 | 4 KiB (= page → VIPT) | cpus 0,32 (SMT siblings) |
| L2 | 1 MiB | 64 | 8 | 2048 | 128 KiB (physically indexed) | cpus 0,32 |
| L3 | 16 MiB | 64 | 16 | 16384 | 1 MiB | cpus 0–3, 32–35 (one CCX: 4 cores, 8 threads); 8 such L3s in all |

NUMA: node 0 = cpus 0–15, 32–47; node 1 = cpus 16–31, 48–63.

Consequences for the current configuration:

* **`SM_NTHREADS: 64` is the logical count, not the physical one** (the comment in `n75_linux.yaml` says the opposite). Every worker therefore shares its L1 and L2 with another worker. Per-thread shares are **L1 16 KiB, L2 512 KiB, L3 2 MiB**.
* **`SM_SLC_BYTES: 64 MiB` matches no cache on this machine.** The largest shared cache is a 16 MiB CCX L3. In `sm_global_ext_merge_par` each of the T merge threads gets `total_streams` buffers of `slc_bytes/(total_streams·16)` entries, i.e. 64 MiB of buffers *per thread*, against a real share of 2 MiB.

  The parameter is therefore acting as an I/O read-ahead size for the SSD/mmap runs, not as a cache budget. That may well be the right value for I/O, but it should be named and tuned as such (`SM_EXT_READ_BYTES`). It should not be derived from the cache size.
* **SMT matters for §3 B3.** Siblings share the VIPT L1, where the set index is the page offset. Two siblings' radix buffers obtained from `malloc`/`mmap` have identical page offsets, so their bucket heads compete for the same sets.
* **NUMA** is a new item (§3 G). Nothing in `ham_dp_c.c` pins threads or places memory, and workers are not tied to the node holding their input slice.
* **Whether 64 threads beat 32** is now a real question for each phase. The sweep is ALU-bound and probably gains from SMT. The radix sort is bandwidth-bound and may lose from halved L1/L2 shares. I'd measure it rather than assume.

## 1. Where the time goes (from the logs, not the code)

Parsed from the `[inst]` lines. Sums of per-step maxima over all steps; the residual is what is not covered by the four phases.

| run | compute | flush (radix sort + run write) | `merge_runs(seq)` | final merge | residual |
|---|---|---|---|---|---|
| Linux n=75 (`profile_75log.txt`, 64 thr, total 5851 s) | 23.0 % | 8.5 % | **51.7 %** | 15.4 % | 1.3 % |
| M2 Pro n=61 (`m2pro/profile_61m2pro.txt`, total 3024 s) | 19.2 % | **31.2 %** | 0.0 % | 27.4 % | — |

Readings (inferences from these numbers; not separately measured):

* **Compute (sweep)** reads one input stream and writes one output stream per thread.

  At step 52 (n=75), each worker does ≈ 36 M inputs × 16 subsets in ≈ 71 s, i.e. ≈ 120 ns per subset evaluation. It writes ≈ 2.2 GB in that time, ≈ 2 GB/s across all 64 threads. That is far below the machine's bandwidth, so this phase is ALU/branch bound in `apply_edge`/`canon`/`apply_elim_seq`.

  Cache geometry will not move it.
* **Flush** is an 8-pass LSD radix sort on `eh_hash(key)`: 8 reads + 8 writes of the buffer, plus 16 hash evaluations per entry.

  On Linux it is ~9 %. On the M2 Pro, with 6–8 threads per far smaller memory system, it is 31 %. This is the phase where line size, L1 associativity and cache sharing really enter.
* **The serial `merge_runs`** typically has n_runs = 2–5 per worker: 64 sequential merges of ~2–3 s each per step.
* **The final merge** is parallel already; the ext variant is I/O bound.

## 2. Gathering the geometry

### 2.1 What the OS tells us

| quantity | Linux | macOS |
|---|---|---|
| size, line, ways, sets per level | `/sys/devices/system/cpu/cpuN/cache/index*/{size,coherency_line_size,ways_of_associativity,number_of_sets}` | `hw.perflevel{0,1}.l1dcachesize`, `hw.perflevel{0,1}.l2cachesize`, `hw.cachelinesize`; **no ways/sets**; no SLC size |
| sharing | `.../index*/shared_cpu_list`, `.../topology/thread_siblings_list` | `hw.perflevel{0,1}.cpusperl2`, `.physicalcpu`, `.logicalcpu` |
| page size / huge pages | `getconf PAGESIZE`, `/sys/kernel/mm/transparent_hugepage/enabled` | 16 KiB on Apple Silicon; no user superpages on arm64 |
| zero-line granularity | — | `DCZID_EL0` (readable from user space) |

Caveats:

* On ARM Linux, sysfs is only as good as the firmware tables. In the Cowork Linux VM on your Mac, `index*/size` and `ways_of_associativity` are **empty**.
* The current `_detect_cache_sizes` reads `hw.l1dcachesize`/`hw.l2cachesize` rather than the `perflevel0` values. On some Apple chips the unqualified key reports a different core type. It also falls back to `hw.cachesize` for L3, which is an array, not the SLC.

### 2.2 What must be measured (new `src/C/cacheprobe.c`)

A small C program, run from `tune_params.py`, that prints YAML.

Every measured value carries `source: measured`, `confidence` (fraction of repeats agreeing with the mode), and the raw latency curve kept in a side file. If the OS value and the measured value disagree, both are recorded and the OS value is used.

1. **Coherence granularity** (the false-sharing unit).
   * Method: two threads hammer counters `d` bytes apart, for d = 8 … 512. Throughput collapses for d < granularity.
   * This detects Zen/Intel's adjacent-line prefetcher pairing (effective 128 B) as well as the 64 B line, and Apple's 128 B.
   * Linux pins the threads; macOS cannot pin, so it runs repeated trials.
2. **L1 associativity and sets** (VIPT, so virtual strides suffice).
   * Method: a pointer chase over k addresses at stride S = 2^j, with k = 1 … 32, in random order to defeat the prefetchers.
   * For S a multiple of the way size W = size/ways, the latency jumps exactly when k > ways.
   * The smallest S that shows the jump at k = ways + 1 gives W, hence sets = W/line.
   * This is deterministic and should be robust.
3. **L2 associativity** (physically indexed, way size > page).
   * Strides do not work here. Instead, use minimal eviction sets by group testing over random lines from a large buffer: Vila, Köpf & Morales, "Theory and practice of finding eviction sets", IEEE S&P 2019.
   * For LRU-like policies, the minimal eviction-set size equals the associativity. Repeat for ~50 targets and take the mode.
   * Caveats:
     * adaptive or non-LRU replacement (Apple's is undocumented) can make the minimal set fuzzy;
     * inclusive vs. victim L3 changes what is measured (Zen 4's L3 is a victim cache of L2);
     * the Linux box had another user during your runs, so the noise could be serious.

   If confidence is < 0.8, record `ways: null`.
4. **SLC / L3** size by the standard working-set latency sweep. Associativity is not attempted: it is hashed or memory-side.
5. **dTLB reach** (optional): a one-access-per-page chase over k pages. This gives the L1/L2 TLB entry counts, which bound the useful radix fan-out.

None of the measurements needs root.

### 2.3 machine.yaml additions

The existing `SM_*` keys keep their meaning. Informational keys stay underscore-prefixed:

```yaml
_cache:
  L1d: {size: 32768, line: 64, ways: 8, sets: 64, shared_by: 2, source: sysfs, confidence: 1.0}
  L2:  {size: 1048576, line: 64, ways: 8, sets: 2048, shared_by: 2, source: sysfs}
  L3:  {size: 16777216, line: 64, ways: 16, sets: 16384, shared_by: 8, instances: 8, source: sysfs}
_coherence_bytes: 128        # measured false-sharing granularity
_page_bytes: 4096
_thp: madvise
_smt_per_core: 2
_dczva_bytes: null           # ARM only
_dtlb: {l1: 72, l2: 3072, source: measured}
```

The Linux values above are now confirmed by sysfs (§0a), and the `_dtlb` numbers are placeholders until measured. On Linux x86, `cacheprobe` will only *verify* sysfs (items 1, 2, 5); it skips the eviction-set measurement unless sysfs is empty.

Add a `_numa` key: `{nodes: 2, cpus: ["0-15,32-47", "16-31,48-63"]}`.

## 3. Using it in `ham_dp_c`, in order of expected value

### A. Make the RAM-path merge parallel *within a memory budget* (not geometry; the ~50 % item on Linux)

Revised after your note that the serial Stage 1 was deliberate (an OOM). The goal is therefore not "parallel instead of serial" but **parallelism chosen from the memory actually available**, with a peak that is never above today's.

**Notation.** Let R be the total size in bytes of all workers' sorted runs at the end of the sweep, and R_i worker i's share, so that R = Σ R_i. Let H be the headroom: the RAM budget (`ram_bytes` − OS reserve) minus what is live at that point.

**Today's peaks.** `curr` is already freed before Stage 1.
* Stage 1 (serial): R + max_i R_i, because worker i's `out` (≤ R_i) coexists with every run.
* Stage 2: W-outputs (≈ R; the per-worker dedup is 1.00–1.06×) + `nxt` allocated at `raw_total` ≈ R. That is ≈ 2R, unless `use_pairwise` kicks in.
* Running Stage 1 fully in parallel would need ≈ 2R at once. That is presumably the OOM you hit.

`sm_should_use_ext` also budgets the RAM path at 2 × raw, so every step whose 2R does not fit goes to SSD.

I propose two steps.

**A0 — parallel inside each worker's merge (zero extra memory).**
* `sm_merge_runs` already splits a worker's runs into 256 independent hash buckets (`bounds[i][b]`). Merge those buckets with T threads. Bucket b writes into `out` at offset Σ_{c<b} (input size of bucket c) — an upper bound, since dedup only shrinks. Then compact left in bucket order. Compaction is cheap: the logs show 1.00–1.06× dedup at this stage, so gaps are a few %.
* Workers stay serial, so the peak is exactly today's R + max_i R_i. The 256-way split gives ample load balance for T = 64.
* Expected: Stage 1 goes from 3026 s to roughly 3026/(0.6·T) ≈ 80 s on the Linux machine. The 0.6 is a guess at the parallel efficiency of a bandwidth-bound merge on 32 cores / 64 threads; to be measured.
* This is the smallest change and the one I'd do first.

**A1 — one global merge with streaming release (lower peak than today, and parallel).**
* Replace Stage 1 + Stage 2 with one splitter-partitioned merge over all P·n_runs RAM runs into `nxt`, the same algorithm as `sm_global_ext_merge_par`. Its stream descriptor is generalised from `{fd, map, off, len}` to `{const SMEntry *base, len}`, so the RAM and ext paths share it.
* `nxt` is reserved as virtual memory of size R. Only written pages become resident.
* Each merge thread *returns the input it has consumed*. It owns a contiguous slice of every run, and after advancing a stream by a release granule G it calls `madvise(MADV_DONTNEED)` (Linux) / `MADV_FREE` (macOS) on the page-aligned interior of what it has passed.
* Resident memory is then (unconsumed input) + (written output) + slack. Written output ≤ consumed input, so

  **peak ≤ R + T·K·G**, where K = number of streams, versus ≈ 2R today. This is a bound from the construction; the madvise cost and the overcommit behaviour are to be measured.
* **This is where parallelism adapts to memory.** Given the headroom H at merge time, choose T ≤ `SM_NTHREADS` and G ≥ 1 page such that T·K·G ≤ H/2. Prefer large T and shrink G down to 64 KiB before reducing T. Fall back to A0 when even T = 1 does not fit.

  For example, K = 192 streams, T = 64 and G = 256 KiB give a slack of 3 GiB.
* Afterwards: compact the T output ranges (each thread wrote at its upper-bound offset), then `realloc` down as now.
* Side benefit: with the peak at ≈ R instead of 2R, `sm_should_use_ext` can budget R + slack. Some steps that now go to SSD would stay in RAM. That is a further, separate speed-up, which I'd enable only after A1 is validated.
* Risks:
  * Linux overcommit policy could refuse the virtual reservation. Mode 0 refuses only reservations larger than RAM + swap, and R is below that. Mode 2 would refuse; I'll check `/proc/sys/vm/overcommit_memory` at start-up and fall back to A0.
  * On macOS `MADV_FREE` is lazy, so resident memory only drops under pressure. That is acceptable, but RSS readings will look higher than the real peak.

**Instrumentation for both.** Log the chosen T and G and the predicted peak, and read the actual peak RSS from `getrusage`/`/proc/self/status` (VmHWM) each step. The memory claim is then checked by measurement, not assumed.

**Correctness.** Dedup is associative and commutative over the runs, so `nxt` is identical. Tests compare the state counts, a checksum of `nxt` per step, and the final counts against the current code (n ≤ 60 here; one n ≈ 70 run by you on Linux).

### B. Thread-interference fixes (cheap)

* **B1. Alignment that actually happens.** `WS = calloc(P, sizeof(SMWorkerState))` only guarantees 16-byte alignment, so the `aligned(64/128)` attribute pads the *size* but not the *base*. Workers' `buf_len` lines can therefore be shared with their neighbours. Fix: `posix_memalign` with `max(_coherence_bytes, 128)`; also apply it to the `SMWorker` array and the merge jobs. Use 128 on x86 too, because of the adjacent-line prefetcher.
* **B2. Per-thread budgets from shares.**
  * Today's formulas:
    * `SM_RBUF_SIZE` takes "all of L2" per thread — on Apple, L2 is shared by a whole cluster;
    * `SM_BB_BUF` uses `P × size ≤ L1`. *Correction to the first version of this note:* here P is the number of streams one merge thread reads (K = P), not the number of threads sharing L1, so the formula was right apart from using the whole L1 instead of the thread's share;
    * `SM_SLC_BYTES` uses one global SLC/L3 figure — on Zen 4, L3 is per-CCD.
  * Replace these with `share(level) = size / (threads actually sharing it)`. "Threads actually sharing" comes from `shared_by` and the thread count; on macOS, assume P-cores only.
  * This is a formula fix in `ham_dp_c.py` and `tune_params.py`. The C code is unaffected.
* **B3. Colouring for SMT siblings.** Hot per-thread buffers that sit in a shared VIPT L1 (SMT siblings on Zen 4) and have identical page offsets compete for the same sets. Fix: offset thread t's radix-buffer base by `(t mod smt) × W/smt`, where W is the L1 way size from §2.2-2. This **does** apply on the Linux machine (§0a: SMT 2, 64 threads on 32 cores). On Zen 4 the L1 way size is W = 4 KiB, so the offset is 2 KiB for odd siblings. C1's L1 budget must also use the per-thread share (16 KiB, not 32 KiB).

The allocator cannot know which sibling a thread will land on unless threads are pinned. So B3 is only fully effective together with G1 (pinning). Without pinning, colouring by worker index still halves the expected collisions, since the odd/even workers placed together are random.

### C. Geometry-aware radix sort (the 31 % item on the Mac, ~9 % on Linux)

* **C1. L1-resident, line-aligned write-combining buffers.** Per bucket, keep `B` lines (B · line/16 entries), with 256 · B · line ≤ ½ · L1 share. The buffer array is one contiguous, line-aligned block.
  * Precise claim: in a VIPT cache with `sets` sets, a contiguous, line-aligned region of R lines puts ⌈R/sets⌉ or ⌊R/sets⌋ lines in each set. So it suffers no self-conflict misses iff ⌈R/sets⌉ ≤ ways, i.e. iff R ≤ capacity in lines. That makes the layout conflict-free *by construction*, so the tuner no longer has to search for it.
  * Why the current layout differs: it uses `rbs = SM_RBUF_SIZE` entries per bucket (64 → 1 KiB stride on Linux; 1024 → 16 KiB stride on the M3, a 4 MiB array). At the start of every pass all 256 bucket heads sit at the same offset mod the stride, so on the M3 they all map to **one** L1 set until the counts drift apart. The tuner's power-of-two sweep cannot see this as a separate effect.
* **C2. Whole-line flushes without read-for-ownership.**
  * x86: non-temporal 16-byte stores (`_mm_stream_si128`) for the full lines of a flush, with ordinary stores for the partial head and tail lines of each bucket's destination range, and an `sfence` at the end of the pass.
  * Apple: `DC ZVA` on each destination line before writing it, if `DCZID_EL0` reports a block equal to the line size. Otherwise ordinary stores; whether Apple cores already elide the read on full-line writes has to be measured.
  * This needs the line size and the ZVA size.
* **C3. MSD first, then sort in cache.**
  * Partition on the top hash bits until each bucket fits in the L2 share. The total fan-out is F = 2^f with (cap/F) · 32 B ≤ L2 share, where 32 B covers the data plus the scratch copy.
  * Each bucket is then LSD-sorted on the remaining bits entirely within L2.
  * The fan-out *per pass* is bounded by the L1 lines available for C1 buffers and by the dTLB reach (§2.2-5); that is where those measurements are used.
  * Linux: cap = 128 M and an L2 share of 1 MiB give F = 4096. That means two partition passes of fan-out 64 plus the in-L2 sort: 3 DRAM round trips instead of 8.
  * M3: cap = 16 M and an L2 share of 16 MiB/4 give F ≥ 128. One pass of fan-out 256 plus the in-L2 sort: 2 round trips.
* **C4. Smaller items.** Hash once per entry per pass instead of twice: cache `eh_hash(key)` in the count pass, or carry the hash in the entry (the finaliser is invertible). Allocate `RadixBuf` once per worker, not once per flush.
* **Expected.** If C1–C3 cut DRAM traffic ~3×, flush goes from 31 % to ~12 % of M2 Pro time and from ~9 % to ~3 % on Linux. This is an estimate; C is gated on a stand-alone sort benchmark showing ≥ 1.5× before it is wired in.

### D. Inconsistencies found on the way (fix regardless)

* `SMEntry` is 16 B (`{u64 key; u64 val}`). However:
  * `ham_dp_c.py` (`SM_ENTRY = 24`, `_wcap_def`, `_rbuf_def`, `_bb_def`, `_ext_def`) assumes 24 B;
  * `tune_params.py` (`ENTRY = 24`, benchmark struct `SME {key, v0, v1}`) does too;
  * so do several C comments (e.g. "P × 2 × worker_cap × 24 B").

  As a result, the analytical defaults under-use memory and cache by 1.5×, and the tuner benchmarks a different record size from the one the DP uses. The EH entry really is 24 B (`u64 + u128`), so the EH formulas are right.
* `_detect_cache_sizes` on macOS: use the `perflevel0` keys (§2.1).

### E. EH path only (low priority; your large runs use sort/merge)

* `EHBkt` with `EH_BKT_CAP = 32` is 784 B (the comment says 3080 B / 128 entries, which is stale). Buckets are not line-aligned, so a key scan touches 3 lines instead of 2. Padding the bucket to a multiple of the line size and aligning the slabs fixes it.
* The `RSORT` slot stride of 640 KiB, with 1024 slots, puts every slot base in the same L1 set and uses 1024 distinct pages per flush.

I'd only do E if you plan to use EH again.

### F. Huge pages on Linux (optional, cheap to try)

`madvise(MADV_HUGEPAGE)` on the large `sm_alloc` regions (input/output tables, worker buffers) cuts dTLB misses in the scatter and merge phases. It also reduces the per-step page-fault cost of re-touching freshly mmapped multi-GB buffers with 64 threads. Note that it also makes L2 set bits virtual-controlled (§0), which is when B3/C1-style colouring would start to matter for L2. It needs THP set to `madvise` or `always` on the server.

### G. NUMA and thread placement (Linux machine; new after `linux_arch.txt`)

* **G0 (no code).** Run one mid-size step range (e.g. n = 70) under `numactl --interleave=all` and once under default first-touch.

  With 2 nodes and unpinned workers, roughly half of all DRAM traffic is currently remote under first-touch: tables written by one thread and read by arbitrary others. Interleaving makes the remote fraction a predictable ½ with both memory controllers used. If it helps, that tells us placement matters before writing any code.
* **G1. Pinning** (`pthread_setaffinity_np`; Linux only — macOS has no affinity API on Apple Silicon). Pin workers so that:
  * the physical cores fill first (worker i → cpu i for i < 32, siblings after), which makes `SM_NTHREADS = 32` versus `64` a clean experiment;
  * a worker's input slice and output buffer live on its node. Partition `curr` by node, so the shard boundaries in the sweep follow the node boundary, and first-touch each worker's buffer from the worker itself (already the case for `buf`/`tmp`).
* **G2. Splitter-partitioned merges by node.** In A and in the ext merge, assign key ranges so that each merge thread writes its range of `nxt` on its own node. The next step's sweep then reads mostly locally if the shards are aligned with those ranges.

G0 is free, so it goes first. G1/G2 only if G0 shows ≥ 10 %.

## 4. Proposed order and acceptance

1. **A0**, then **A1** (memory-adaptive merge). Accept if counts, state counts and per-step `nxt` checksums are identical to the current code for n ≤ 60 (VM tests) and on one Linux run you do at n ≈ 70; if the RAM-step time drops; and if the measured peak RSS is ≤ today's (A0) or ≤ R + slack (A1). You run the Linux timing; I can only do the correctness tests.
2. **B1, B2, D** together. These are small, and D changes defaults only when `machine.yaml` does not override them.
3. **`cacheprobe.c` + `tune_params.py`** integration writing the §2.3 keys, verified against sysfs on Linux and against the published Apple/Zen 4 values where known.
4. **C**, behind a runtime switch (`SM_SORT=lsd|msd`), first as a stand-alone benchmark, then wired in if ≥ 1.5× on the flush phase on your Mac.
5. **B3, E, F** only if the data from 3 says they matter. B3 is now known to be relevant on the Linux machine, but it pays only with pinning (G1).
6. **G0** (an `numactl` run, no code) whenever convenient, and an `SM_NTHREADS` 32-vs-64 comparison at the same time. G1/G2 depend on the result.

## 5. What I can't do from here, and questions

* The Cowork VM on your Mac is a 4-vCPU Linux guest. Its sysfs cache entries are empty, and stage-2 translation perturbs physical indexing. It is fine for correctness tests but useless for tuning. Mac measurements need to be run natively by you (`uv run python -m hamiltonian.claude.tune_params`), and so do the Linux ones.
* ~~Please send `lscpu` and the sysfs cache dump.~~ Done: `data/linux_arch.txt`, summarised in §0a.
* ~~Where does `cacheprobe.c` go?~~ `src/C/cacheprobe.c` (decided).
* ~~Was the serial Stage-1 loop deliberate?~~ Yes, it was there to cap memory. Hence the budgeted design in A.


## 6. Implementation (2026-11-25)

### What changed

| item | file | change |
|---|---|---|
| A0 | `ham_dp_c.c` (`sm_merge_runs`, RAM branch) | A worker's runs are split into NB = 2^lg hash buckets (lg ≥ 8, NB ≥ 16·T, ≤ 4096). T threads claim buckets dynamically and write bucket b at its upper-bound offset; the result is then compacted in bucket order. Workers are still merged one after another, so the peak is unchanged. |
| A1 | `ham_dp_c.c` (`sm_a1_plan`, `sm_global_ram_merge`) | Replaces Stage 1 + Stage 2 when a plan fits the budget: one global splitter-partitioned merge of all K runs; consumed input is released with `madvise`; output goes to per-thread chunks (C ≤ 32 MiB); `nxt` is then allocated at the exact size and filled by a parallel copy. If no T ≥ 1 fits, it falls back to A0 + the existing parallel/pairwise merge. |
| A1 instrumentation | `ham_dp_c.c` | `[inst] ram_merge:` line per RAM step with mode, T, K, G, C, R, budget, predicted peak and the measured peak. The measured peak is VmHWM since the start of the merge (Linux; reset via `/proc/self/clear_refs`), or lifetime `ru_maxrss` elsewhere, labelled as such. |
| test hooks | `ham_dp_c.c` | `HAM_SM_CKSUM=1` prints a per-step digest of the state table, with a sortedness check. `HAM_SM_MERGE=a0\|a1\|auto` and `HAM_SM_MERGE_BUDGET=<bytes>` are overrides for testing. None of them changes default behaviour. |
| B1 | `ham_dp_c.c` | `SM_CACHE_ALIGN` is 128 B on every platform (`SM_ALIGN_BYTES`). `W`, `WS`, the A1 job array and each A1 thread's stream cursors are allocated with `posix_memalign` (`sm_aligned_calloc`). `SMWorker` is now aligned too. |
| B2 | `ham_dp_c.py`, `tune_params.py` | `_detect_cache_sharing()` (sysfs `shared_cpu_list`; macOS `hw.perflevel0.cpusperl2`) and `_per_thread_share()`. The `SM_RBUF_SIZE` default uses the per-thread L2 share and `SM_BB_BUF` the per-thread L1 share. `SM_SLC_BYTES`/`SM_EXT_STREAM_BUF` are left alone (they act as an I/O read-ahead size; see §0a). These are defaults only: keys in `machine.yaml` still win. |
| D | `ham_dp_c.py`, `tune_params.py`, comments in `ham_dp_c.c` | Entry size 24 → 16 B (`SMEntry = {u64 key; u64 val}`). The tuner's benchmark struct now has the real layout. macOS cache sizes come from the `hw.perflevel0.*` keys. |
| probe | `src/C/cacheprobe.c`, `tune_params.py` | The probe from §2.2. `tune_params.py` runs it when writing `machine.yaml` (`--no-probe` to skip) or alone with `--probe-only` (it updates the informational keys of an existing `machine.yaml`). New keys: `_cache`, `_share`, `_page_bytes`, `_coherence_bytes`, `_dczva_bytes`, `_dtlb_knees_pages`, `_latency_knees_bytes` and the raw `_cacheprobe`. |
| tests | `tests/test_sm_merge.py` | For n = 36, 41, checks that A0, A1, A1 under a tight budget and the ext path give identical per-step digests and counts, equal to the EH backend's count. |

### Deviations from §3

* **A1 output.** The plan had output written into a virtual reservation of size R and compacted afterwards. That needs overcommit, and the compaction is serial. Instead, each thread writes into private chunks, and `nxt` is allocated exactly once N is known and filled by T parallel `memcpy`s, freeing each chunk as it goes. The peak bound becomes
  * merge phase: R + min(R, T·K·(G + 2·page) + R_small) + T·C;
  * copy phase: N + T·C.

  Here R_small is the bytes of runs too small to release safely (below). No overcommit is needed.
* **Which runs are released.** Only whole pages inside a thread's own slice of a run are released, and only for runs that are certainly a private mmap:
  * macOS: ≥ 1 MiB, since `sm_alloc` uses mmap there;
  * Linux: ≥ 32 MiB, glibc's maximum mmap threshold.

  Smaller runs count as R_small. On Linux the release is `MADV_DONTNEED`; on macOS it is `MADV_FREE_REUSABLE` (falling back to `MADV_FREE`).
* **Choosing T and G.** Start from `SM_NTHREADS` (1 if R is below `SM_PAR_MERGE_THRESH`) and halve T until the per-stream allowance (budget − R − R_small − T·C)/(T·K) − 2 pages is at least max(64 KiB, page). G is capped at 64 MiB. The budget is `ram_bytes` − 4 GiB, the same OS reserve the rest of the file uses.
* **`_DEFAULT_SOURCE` on Linux.** Under strict `_POSIX_C_SOURCE`, glibc hides `madvise`/`MADV_*`, and `posix_madvise(POSIX_MADV_DONTNEED)` is a no-op. Side effect: the EH path's existing `#if defined(MADV_FREE)` block in `rsort_flush_reset` now compiles on Linux. Until now it had been silently compiled out there, so its comment's intent now actually happens. macOS is unaffected.
* **Alignment** is the constant 128 B, not `_coherence_bytes`. The probe's coherence measurement is too fragile (see below) to drive a compile-time layout.
* **`sm_should_use_ext` is unchanged.** It still budgets the RAM path at 2 × raw. Lowering it to the A1 bound would move some steps from SSD to RAM, but I have left that for after you have validated A1 on the Linux machine.

### Test results (cloud container: 2 vCPU x86 VM, 7.7 GB; not representative for timing)

Per-step digests of the whole state table were identical to the unmodified code in every configuration tried:
* n = 45, 47, 48;
* `SM_WORKER_CAP` 64 K, 2 M and 4 M entries;
* `SM_NTHREADS` 1 and 4;
* `HAM_SM_MERGE` a0, a1 and auto;
* budgets of 0.1, 0.6, 1.0 GB and the default;
* and the ext path.

The final counts were also identical (n=47: 68,361,609 paths; n=48: 140,571,720).

Memory at the same steps (n = 47, `SM_WORKER_CAP` = 4 M, 4 threads; peak during the merge, including ~0.1 GB of Python):

| step | R (runs) | A0 peak | A1 peak (G ≈ 4–15 MiB) |
|---|---|---|---|
| 33 | 0.23 GB | 0.71 GB | 0.45 GB |
| 34 | 0.34 GB | 1.00 GB | 0.59 GB |
| 35 | 0.39 GB | 1.07 GB | 0.63 GB |
| 36 | 0.52 GB | 1.45 GB | 0.65 GB |

So A1's peak is ≈ R + baseline, against ≈ 2R + baseline for A0 (and for the old code, whose peak A0 does not change).

Time on 2 vCPUs:
* the per-worker merge stage at n = 47 (64 K cap) went from 16.4 s serial to 9.4 s with A0;
* A1's single merge took 11.5 s, against 20.1 s for old stages 1 + 2.

These are weak evidence of speed on a 2-core VM. The 32-core Linux machine is where the serial stage costs 52 %, and that measurement is yours to make.

### `cacheprobe` status

Validated only in VMs; it needs a native run on the Mac and on the Linux machine.
* **L1** (VIPT stride method): matched sysfs exactly in the cloud VM (12 ways, 4 KiB way, 64 sets). The first version read the plateau at the largest strides and got 6 ways. The cause was a second plateau from dTLB set conflicts at strides ≥ 64 KiB, so the probe now uses the *first* plateau and prints the whole k\*(stride) curve. In the Linux VM on your Mac (4 KiB guest pages; vCPU placement unknown), results vary between runs:
  * one run's k\* curve was not monotone in the stride, which the probe now detects and reports as `null` (instead of the bogus "1 way" it first produced);
  * another gave 8 ways × 8 KiB. That would match a 64 KiB E-core L1, not the P-core's 128 KiB, so the vCPU may have been on an E-core.

  Treat VM numbers as untrustworthy. That VM did report the DC ZVA block: 64 B.
* **Coherence granularity**: inconclusive in the VM. There was no slowdown even with both counters in one line, so the two vCPUs were evidently not running concurrently on distinct cores. The probe reports `null` in that case, never a guess.
* **L2 ways** (minimal thrashing set): in the VM the reduction stalls at ~570 lines, so it found no small eviction set. The probe reports `null` whenever the minimal set exceeds 65 lines. Whether it works natively, on Zen 4 with 4 KiB pages and on Apple with 16 KiB pages, is untested.
* **Latency and dTLB curves** look plausible. The knees are reported as measured.

### To run on your machines

```
uv run python -m hamiltonian.claude.tune_params --probe-only --output hamiltonian/claude/machine.yaml   # just the probe
uv run pytest tests/test_sm_merge.py
```

and an n ≈ 70 run on the Linux machine with `instrument: true`. The `[inst] ram_merge:` lines then give the chosen T/G, the predicted peak and the measured peak for every RAM step. `HAM_SM_MERGE=a0` reproduces the old memory behaviour with the parallel per-worker merge, for an A/B comparison.
