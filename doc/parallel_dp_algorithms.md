# The parallel frontier DP for H(n): algorithms extracted from the code

**What I read:**
- `hamiltonian/claude/ham_dp_c.c` (April 22 version, 3.3k lines): the authoritative source, especially `count_ham_paths_sm` and its helpers, lines 1266–3330.
- `hamiltonian/claude/ham_dp_c.py` (the cffi wrapper);
- `machine.yaml`, `config/n75_linux.yaml`;
- the session log `doc/new_claude.md`;
- the code snapshots in `doc/Claude/*/files.zip`;
- the run logs `data/linux/profile_*.txt`.

Line numbers refer to `ham_dp_c.c`. Statements about *why* something was done come from `new_claude.md`. Statements about *what* is done come from the code.

---

## 0. Summary

There are two backends.

- **EH backend** (`count_ham_paths_c`). Single-threaded. The table is an extendible hash table, fed through a radix-sort insertion buffer. This is the version described in `talk2.tex` (n = 61 in 97 min, under 2 GB, on an M3 Pro).
- **Sort-merge (SM) backend** (`count_ham_paths_sm`). Multi-threaded, with pthreads. It replaces "hash-insert each contribution" by **generate → sort → merge-with-dedup**. All n ≥ 62 values, and the reruns of n = 60–75 in `data/linux/`, were produced with it. It is not described in `hampath_paper_full.tex` or in `talk2.tex`.

The central observation behind the SM backend is that the contributions to the next table are independent. Each source state `(key, count)` together with each subset S of the back-edges of the new vertex yields at most one pair `(key', count)`. Only the **reduction** (summing counts with equal `key'`) couples different sources. So:

1. **Generate in parallel.** Each worker takes a contiguous shard of the current table and emits pairs into a private buffer.
2. **Reduce by sorting.** Each worker sorts its buffer by a hash of the key and deduplicates it into a sorted *run*. Runs are kept in RAM or spilled to SSD.
3. **Merge in parallel.** All runs are merged with deduplication by P threads, each owning a disjoint range of *hash space*.

Nothing is shared and written during either phase. There are no atomics on counts and no locks except one mutex for the final scalar total.

---

## 1. State encoding (unchanged from the serial DP)

A state is a `u64` key (lines 166–255).

- Frontier slot i occupies bits 4i … 4i+3 and holds `enc_v(v) = v + 2`, where:
  - v = 0 means the vertex has no edges yet;
  - v = −1 means interior;
  - v = L > 0 means an endpoint of chain L.
- Bit 62 (`KEY_NC_BIT`) is the closed-chain flag c. Bit 63 is a marker bit.
- So at most **15 frontier slots** fit (4·15 + 2 = 62 bits). Exceeding this aborts (`MAX_FS_FAST = 15`).
- `canon` relabels chains in order of first appearance.

**An SM entry** (`SMEntry`, line 1289) is `{u64 key; u64 val}`, 16 bytes. The count was u128 (24 bytes) until it was reduced to u64 to save 33% memory (see `new_claude.md`, around line 2409). Some code comments still say "24B". Overflow handling:
- every addition of two counts goes through `SM_VAL_ADD` (lines 39–46), which sets a global flag on wrap-around;
- the Python wrapper raises `OverflowError` if the flag is set (`ham_dp_c.py`, line 671);
- the final total and the cycle count are accumulated in u128.

---

## 2. One DP step in the SM backend

```
for step t = 0 … n−1:                       # vertex v_t of the ordering
    A. INTRODUCE (in place, parallel)        # sm_introduce, l.1711
         for each entry: key |= enc_v(0) << (4·fs)          # new slot for v_t
         (only ORs new high bits, so entries are independent; no copy of the
          table → saves a whole table, 303 GB at n=75 step 57 per the comment)
    B+C. FUSED SWEEP (parallel)              # sm_fused_sweep, l.2872
         decide ext (disk-spill) vs RAM mode            # sm_should_use_ext
         launch P workers on static shards of curr       # sm_worker_runs
         merge all workers' runs into nxt                # RAM path or ext path
    free curr; curr ← nxt; update frontier
```

### 2.1 Worker (generation + local reduction), `sm_worker_runs` (line 2165)

```
input: shard curr[lo:hi), lo = i·chunk, chunk = ⌈|curr|/P⌉
buf ← empty (capacity CAP = worker_cap entries)
for each (key, cnt) in shard:
    for each subset S ⊆ {back-edges of v_t}  (2^n_back subsets, bitmask loop):
        k' ← key; apply each edge of S with apply_edge     (reject: degree>2, cycle)
        [last step only, optional: detect Hamiltonian cycles, add to local_cycles]
        apply the eliminations of this step (apply_elim_seq); a state that completes
            the single chain adds cnt to local_total (u128) and is not emitted
        append (k', cnt) to buf
        if buf full: FLUSH-RUN
FLUSH-RUN on the remainder; add local_total to the global total under a mutex
```

`FLUSH-RUN` (`sm_flush_run`, line 1971):
1. Sort `buf` by `eh_hash(key)` with an 8-pass, buffered, 8-bit LSD radix sort (`sm_radix_sort`, line 1349).
2. Deduplicate adjacent equal keys in place, adding counts with `SM_VAL_ADD`.
3. Store the result as a **run**, either in RAM (`sm_alloc`) or appended to the worker's backing file (`ext` mode).

**Memory bound.** Peak RAM during generation is `|curr| + P·2·CAP·16 B` (buffer plus radix scratch per worker), *independent of the output size*. This is what fixed the out-of-memory failure at n = 61, step 37 and later.

### 2.2 RAM or disk: `sm_should_use_ext` (line 2855)

```
chunk = ⌈si/P⌉;  bpt = chunk·2^nb               # raw outputs per worker (upper bound)
n_runs = ⌈bpt/CAP⌉;  raw_est = n_runs·CAP·P·16 B
merge_peak = 2·raw_est                          # RAM-path merge holds inputs + nxt
avail = RAM − 4 GB(OS) − |curr| − P·2·CAP·16 B
ext  ⇔  merge_peak > avail
```

In ext mode each worker's runs go to an anonymous mmap or shared-memory region sized for the worst case. Pages are committed lazily. An earlier, smaller size estimate caused silent data loss, which the comment at line 2885 documents.

### 2.3 Merge

There are two paths, chosen per step.

**RAM path.**
1. `sm_merge_runs` (line 2038), one call per worker, merges that worker's runs. It is **bucketed by the top byte of `eh_hash`**: 256 independent sub-merges, each small enough to stay in L2.
2. The P per-worker streams are then merged into `nxt`. Normally this uses `sm_parallel_merge` (line 1576); if memory is tight it uses `sm_pairwise_merge_into` (line 2271), a binary-tree reduction that frees inputs as it goes.

**Ext path** (large steps). `sm_global_ext_merge_par` (line 2471) merges all K = Σ n_runs streams, from all workers, straight from SSD into `nxt` in a single pass. At n = 75, K reached 694. Each stream is read through a buffer of about `SLC_bytes / K` entries, the "Aiken capacity-boundary" sizing, clamped to `[ext_stream_buf, 1 MB]`.

**Parallel K-way merge, used in both paths:**
```
1. sample 64 hash values per stream; sort samples; choose T−1 splitters
   that cut hash space into T pieces with ≈ equal sample mass
2. binary-search every stream for every splitter (streams are hash-sorted)
3. upper-bound each range's output size; allocate nxt; give range r a slice
4. T threads: thread r heap-merges its K sub-streams with dedup into its slice
5. compact: prefix-sum the actual counts, memmove to close gaps
```
Correctness rests on one fact: equal keys have equal hashes, so they fall in the same range. Dedup is therefore exact within each range, and no two threads ever produce the same key.

---

## 3. Load balancing: what was actually done

**(a) Why the work is hard to predict.** The work generated by a source state is the number of subsets S that survive `apply_edge` and the eliminations. It depends on the state's degree and chain pattern, and it varies by orders of magnitude between states. Nothing cheap computed from the key predicts it well, which is the point you made.

**(b) The solution in the code: hash order makes static shards statistically balanced.** After each merge, `nxt` is sorted by `eh_hash(key)`, i.e. by a bijective mixing of the key. The in-place introduce at the next step changes every key but moves no entry, so the table is then ordered by the hash of the *previous* key, which is equally unrelated to the structure of the new states. So a contiguous shard `[i·chunk, (i+1)·chunk)` of `curr` is effectively a **random sample** of states, uncorrelated with their structure, and each worker receives about the same distribution of work. Measured compute balance on the n = 75 run (64 workers), as max worker time / mean worker time:

| step | 30 | 36 | 38 | 39 | 42 | 46 | 49 | 52 | 56 | 58 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| max/mean | 1.03 | 1.19 | 1.09 | 1.12 | 1.17 | 1.18 | 1.07 | 1.12 | 1.17 | 1.11 | 1.13 |

That is within 20% on the heavy steps, which dominate the runtime. On light steps it is worse (1.5–1.9 on steps 43, 47, 54, 55 and 64), but those steps are short.

**(c) The same idea balances the merge, and its absence was a real bug.** Raw keys are very non-uniform: the top byte holds only flag bits. An earlier version sorted runs by the **raw key** but chose merge splitters in **hash** space. The splitter ranges were then meaningless, and 5 of 6 merge threads sat idle (about 10% CPU at n = 61). The bucketed merge had wrong bucket boundaries too. The fix was to sort everywhere by `eh_hash(key)`: in the radix passes, the heap comparisons, `sm_bucket` and `sm_lower_bound`. Hash order then gives near-perfect 256-way buckets and balanced splitters (`new_claude.md`, lines 2080–2100).

**(d) Work stealing was designed but never used.** A shared micro-chunk queue (`SMWorkQueue`, `sm_wq_claim`, line 1939, with `SM_STEAL_FACTOR = 32` micro-chunks per worker) is defined in this version and in every snapshot from the Seventeenth session on. **No snapshot ever calls it.** The configuration still carries `steal_factor`, but it has no effect. If balance on the light steps matters, wiring it in would mean replacing `for i in [lo,hi)` with a loop that claims micro-chunks. It would not change the memory bound.

**(e) False sharing.** The per-worker state is aligned to 128-byte cache lines (64 bytes on x86). Without this, writes to `buf_len` by different workers invalidated each other's cache lines, causing a 3–4× slowdown on steps with 2 runs (line 1762).

---

## 4. Configuration and tuning

`tune_params.py` derives the runtime constants (passed in via `sm_configure`) from the cache sizes and core count of the machine:

| parameter | meaning | M3 (machine.yaml) |
|---|---|---|
| `SM_NTHREADS` | P, the number of workers and merge threads | 4 |
| `SM_WORKER_CAP` | entries per worker buffer before a run is flushed | 16M |
| `SM_RBUF_SIZE` | radix-sort slot buffer | 1024 |
| `SM_BB_BUF` | block buffer of the K-way merge (one DRAM miss per block) | 1024 |
| `SM_EXT_STREAM_BUF` | lower bound on the stream buffer in ext merges | 8192 |
| `SM_PAR_MERGE_THRESH` | below this size, merge serially | 50M |
| `SM_SLC_BYTES` | last-level cache size, used to size stream buffers | 12 MB |
| `SM_STEAL_FACTOR` | (unused, see §3d) | 32 |

The orderings come from the MaxSAT exact-pathwidth solver followed by simulated annealing (`ham_ordering.py`, `sa_cost.py`). For SM, the cost model was extended because a spilled step costs 2–3× a RAM step of the same size (`new_claude.md`, around line 2542).

---

## 5. Data from the runs (`data/linux/`)

- **n = 74.** Four complete runs give **H(74) = 2 176 025 661 130 535 741**. All four use the *same* ordering (identical step profile and work). One used the April 13 code (3449 s); three used the code of April 16–17 (945, 957, 1205 s). They cross-check the code changes, not the ordering.
- **n = 75.** **One** complete run, `profile_75log.txt`:
  - width 15, the maximum the key encoding allows;
  - ordering time 2179 s (of which MaxSAT 1284 s), DP time 5851 s;
  - 64 workers;
  - peak table of 3.8·10⁹ states going into a step; up to 1.2·10¹⁰ raw contributions per step before dedup (step 38, dedup factor 3.4);
  - 8 steps spilled to disk, with up to 694 runs;
  - run on the 768 GB, 64-core Linux machine, with `machine.yaml` hand-tuned until the run completed;
  - result `ham_paths = 652211775281872067084`.
  - **Withdrawn (2026-11-27).** This run used the old 15-slot key, in which the in-step frontier of 16 vertices at width 15 overlapped the flag bits. With the fixed 16-slot key (`ham_dp_c.c`), two runs with different optimal orderings (work model and v7) both give 9231885593047053408 (`data/linux_data/new_75_sm_new_pruned*.txt`).
- The other 13 `profile_75*` logs are aborted attempts that ran out of storage, several of them after reaching width-15 steps. None of them produced a total, so they say nothing about the correctness of the width-15 path. Every completed run with n ≤ 74 has max width 14.

### Why n = 74 was faster than n = 62–73

All of the runs for n = 62–73, and the first n = 74 run (`profile_74linux.txt`), used the code of April 11–13. That code had 24-byte entries (u128 counts) and re-sorted the whole table by `eh_hash` after every introduce. The later n = 74 runs and the n = 75 run used the code of April 16 onward, with 16-byte entries and the in-place introduce. All four n = 74 runs did identical work: Σ states_in·2^nb = 1.31·10¹¹. Totals over all steps (s):

| run | DP | introduce | generate (max worker) | flush (max worker) | merge | spilled steps |
|---|---|---|---|---|---|---|
| `74linux` (Apr 13) | 3449 | 1417 | 345 | 581 | 1002 | 3 |
| `74mlp_linux` (Apr 16) | 945 | 12 | 337 | 306 | 240 | 0 |
| `74new` (Apr 17) | 957 | 4 | 496 | 289 | 237 | 0 |

- **Introduce.** Removing the post-introduce sort saved about 1400 s. The sort was never needed: the introduce is injective on keys, and workers do not need sorted input. In the old runs for n = 62–74 it was 40–50% of DP time.
- **Merge.** About 760 s, mostly because with 16-byte entries the three heavy n_back = 4 steps (35, 49, 55) no longer spill to SSD. Step 35 alone went from 433 s to 56 s.
- **Flush.** About 280 s, from moving a third fewer bytes.
- **Generation.** Unchanged (345 s vs 337 s). Generation is the CPU-bound part, so it is where competition from another user would show up most. It did not change, so contention does not explain the difference. In `74new` generation is *higher* (496 s), and the slowest-to-mean worker ratio is worse, which may be that kind of interference.

Normalised per unit of work, the old runs for n = 62–74 all take 21–28 ns per (state × subset), and the new n = 74 runs take 7–9. So the old n = 62–73 times would presumably drop by about 3× with the current code.

---

## 6. Why H(75) deserves a recheck: two first-time boundary conditions

The scaling predictor in the paper puts H(75) about e^{4.3} ≈ 70 times lower than the reported value. Two code paths are exercised for the first time at exactly n = 75:

1. **Frontier width 15 = the encoding maximum.** Slot 14 occupies bits 56–59, adjacent to the flag bits 62–63, and steps 42 and 48 of the n = 75 run have fs = 15. An older comment (line 2015) says that "bits 56–61 are always zero". That was true up to width 14 and is no longer true at width 15. I found no code that relies on it (bucketing hashes the key first), but **no completed run before n = 75 ever had width 15**.
2. **The total exceeds 2⁶⁴.** H(74) ≈ 2.2·10¹⁸ < 2⁶⁴ ≈ 1.8·10¹⁹ < H(75). Totals are u128 and per-state counts are u64 with an overflow flag, so this path is guarded, but it was never exercised before n = 75.

**Cheap tests that need no big machine:**
- **(a) The width-15 path.** Run the SM backend on n ≈ 40–55 with a deliberately poor ordering whose frontier reaches width 15, and compare with the known counts. This exercises slot 14 and the flag bits at small cost.
- **(b) The u64/u128 boundary.** Rerun a known n with the counts artificially scaled, e.g. start the table with count 2⁴⁰ instead of 1, and check that the result is exactly 2⁴⁰·H(n), or that overflow is reported.
- **(c) If a large machine becomes available.** Rerun n = 75 with a different optimal ordering. (No n ≤ 74 has been cross-checked across orderings either; the four n = 74 runs share one ordering.)

---

## 7. What to change in `talk2.tex` and `hampath_paper_full.tex`

- **`hampath_paper_full.tex`** says nothing about threads. Its record of earlier attempts ("parallelism abandoned because the serial merge negated the gains") is now obsolete. It needs a section on the SM backend (§§2–3 above) and results to n = 75, with the caveat about H(75).
- **`talk2.tex`** ends at n = 61 with the serial EH backend. Natural additions:
  - a part on "generate → sort → merge", explaining why merging sorted runs scales where hash-insertion did not (no shared writes; dedup is local in hash ranges);
  - the hash-order balancing trick, together with the raw-key bug;
  - the RAM/SSD decision and the ext merge of about 700 streams;
  - the n = 74/75 numbers and the open recheck of n = 75.
- **Machine.** 768 GB and 64 cores (confirmed). `app_algorithms.tex` now says this, and its `%% CHECK` sentence has been replaced by a paragraph on the sort-merge parallelisation.
