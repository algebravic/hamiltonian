# Sort-merge DP: in-sweep spill decision and CRT snapshots — design

Status (2026-11-25): **implemented**; see §8.
Context: the n = 75 run with the kpw ordering (`data/linux/new_75_sm_new.txt`). It was at step 49 after 1 723 s; April's run took 3 434 s to reach step 49.

## 0. A correction first: "ext" steps are not on SSD

Reading `sm_ext_open` closely: the ext backing store is first tried as **`mmap(MAP_PRIVATE | MAP_ANONYMOUS)`**, i.e. anonymous memory. It is RAM, or swap if the machine has swap. Only if that reservation is refused does it fall back to `tmpfile()`, i.e. disk (tmpfs or a real filesystem). The reservation is per worker, chunk × 2^n_back × 16 B, e.g. ≈ 13 GB at step 44.

On the Linux machine, an "ext" step therefore keeps its runs in RAM as well. What it differs in is:

| | "sm" (RAM) steps | "ext" steps |
|---|---|---|
| run storage | one `sm_alloc` per run, plus a `memcpy` from the worker buffer | one anonymous mapping per worker, plus a `memcpy` into it |
| merge | A1: global merge, can release consumed input | `sm_global_ext_merge_par`: buffered `memcpy` per stream, never releases input |
| output | private chunks, then an exact-size `nxt` | `nxt` reserved at raw size, filled, then `realloc`'d down |
| peak during the merge | ≤ R + N, and → R when memory is tight | R + N + read buffers |
| merge cost at n = 75 (ns per raw entry) | 3.5 (step 43) | 6.0–7.4 (steps 40–44) |

My earlier statement that these steps "spill to SSD" was wrong. The conclusion stands, though: steps 40, 41, 42 and 44 took 694 s on the slower path, chosen by a pre-sweep estimate that was 3× too pessimistic. They would all have fit in A1's budget. Whether the April failures ("running out of storage") were swap, the `tmpfile` fallback or the filesystem, I can't tell from here. Does the Linux machine have swap, and where does `TMPDIR` point?

## 1. Goals

1. Every step takes the fast path (RAM runs + A1) whenever its memory actually fits, judged from measured sizes, not from a pre-sweep estimate.
2. When memory genuinely does not fit, spill to **real disk**, and only the part that does not fit, in a predictable place. Today the system silently falls back to swap or `tmpfile`.
3. Never exceed the budget by construction, and log the predicted and measured peaks per step.
4. For the CRT, keep a snapshot from which a replay is always possible, without a full rerun, at a cost proportional to what is actually needed.

## 2. One run layout, decided per run during the sweep (replaces `sm_should_use_ext`)

**Accounting.** One process-wide atomic counter `resident_runs` holds the bytes of all RAM runs of the current step. The fixed part during the sweep is

  F = curr (s_in · 16 B) + worker buffers (P · 2 · cap · 16 B) + the held snapshot (§4) + 4 GiB OS reserve,

and the merge reserve is M_res = T · K_est · (G_min + 2 pages) + T · C, the smallest A1 plan, with K_est = runs so far + P.

**Rule at each flush** (a worker's buffer is full and has just been sorted and deduplicated to b bytes): atomically add b to `resident_runs`. If the result exceeds RAM − F − M_res, take the b back off and **write this run to the spill file** instead (§3). Otherwise keep it in RAM as now.

**Why the bound holds.**
* *Sweep.* Resident ≤ F + `resident_runs` ≤ RAM − M_res.
* *Merge.* Afterwards `curr` and the buffers are freed, so A1 starts with `resident_runs` + snapshot + reserve and at least M_res free. A1's own plan then picks T and G within what is free; that is already implemented and was measured at steps 38–43. If the actual run count is larger than K_est, A1 lowers T (the existing halving loop), and in the limit falls back to fewer threads.

No estimate of the step's output is involved. The decision is made per run, by the worker that produced it.

**Retired.** The pre-sweep `sm_should_use_ext` and the anonymous-mapping ext store are no longer used for memory decisions. I'd keep `sm_global_ext_merge_par` for one release behind `HAM_SM_LAYOUT=ext`, for A/B comparison.

## 3. Real disk spill: mixed streams in A1

* **Spill store.** One file per worker in `SM_SPILL_DIR` (default `$TMPDIR`, then `/tmp`). It is created with `mkstemp` and immediately `unlink`ed, so nothing is left behind even after a crash. Runs are appended with `pwrite` of the sorted, deduplicated block.
* **Merge.** A1 gets a second stream kind, `{fd, off, len, buf}`, read through a per-stream buffer of B_f bytes (default 8 MiB, sequential `pread`) followed by `posix_fadvise(POSIX_FADV_DONTNEED)`, so the page cache does not compete with the tables. The splitter sampling for file streams uses `pread` of single entries. The binary searches for slice bounds use `pread` too: O(log len) small reads per stream per thread, which is negligible.
* **Merge budget.** The file streams add T · K_file · B_f to the plan's fixed part. If that does not fit, B_f shrinks to 1 MiB before T is reduced.
* **Release** applies to the RAM streams only, as now. The spill files are closed at the end of the merge.
* **Cost model.** A spilled run is written once and read once, sequentially. At a few GB/s this is comparable to the merge itself, so spilling is a real cost, and it only happens when RAM is really short.

## 4. CRT snapshots

The rolling ≤ 1 GiB snapshot works when the first wrap comes late (n = 74: step 68, 48 M states). At n = 75 the tables are much bigger, so a wrap in a big step would force a full rerun. Two additions:

**4a. A rigorous "no wrap possible" test per step.** Every output count of step s is a sum of contributions, each contribution is the count of one input state, and each input contributes to at most 2^{n_back} outputs. Hence

  every count after step s ≤ U_s := 2^{n_back(s)} · Σ_{inputs} count,

with Σ computed exactly in `u128`. This is cheap: one extra add per entry, inside the parallel `sm_introduce` pass. If U_s < 2⁶⁴, step s **cannot** wrap. The bound is very loose but sufficient for this purpose, and Σ and U_s go into the `[inst]` line, so we will learn where the risk starts.

**4b. Snapshot only where it can be needed.**
* **Anchor snapshot.** At the start of the first step s* with U_{s*} ≥ 2⁶⁴, the table is still exact, so take one snapshot of the table and the frontier state.
  * If it fits in `crt_snap_bytes` it is kept in RAM, and counted in F above. The new default is min(RAM/16, 64 GiB): 47 GB, about 2.9 G states, on the 768 GB machine; 1 GB on a 16 GB Mac.
  * Otherwise it is written once to `SM_SPILL_DIR`. For a 7 G-state table that is 112 GB of sequential writing, roughly 1–2 minutes on NVMe. It is read back only if a wrap actually happens.
* **Rolling snapshots** after s*, as now, for steps whose input fits the rolling budget (≤ 1 GiB), until the first actual wrap. A wrap late in the run is then replayed from a small table.
* **Replay** starts from the latest snapshot at or before the first wrap step. The worst case is a replay from s*, which never costs more than a full rerun and usually much less, since s* comes at or after the point where the counts first become large.
* **No wrap at all.** Then the only cost is the anchor copy (RAM or one disk write) and the rolling `memcpy`s, all after s*.

**Options.** `ham_count --crt-snap-gb X` sets the RAM limit for the anchor (0 forces it to disk). `--spill-dir PATH` applies to both spill and snapshot files. They are exposed in `count_hamiltonian_paths_sm` as `crt_snap_bytes=` and `spill_dir=`.

## 5. Instrumentation added

On each step's `[inst]` lines:
* `resident_runs` at the end of the sweep, and the number of spilled runs and bytes;
* U_s (as log₂) and whether a snapshot was taken (anchor, rolling or none);
* A1's line as now, with K split into RAM and file streams;
* `n_runs` printed before the runs are freed (fixes the `n_runs=0` cosmetic bug).

## 6. Tests

* **Spill.** A test hook forces a small budget (`HAM_SM_MERGE_BUDGET` extended to the sweep rule) so that some runs spill. Per-step digests must equal the all-RAM run at n = 41 and 45. Variants: no spill, partial spill, everything spilled.
* **U_s.** At n = 41 with `_word_bits` = 12, 16 and 20, check that no wrap ever occurs at a step with U_s < 2^b. The 2^b analogue of the test is exact, so this is a direct check of the inequality. The CRT tests are repeated with the anchor snapshot in RAM, on disk, and absent (falling back to a full rerun).
* **Regression.** The existing `test_sm_merge.py` and `test_sm_crt.py`.

## 7. What I expect, and what I can't promise

* On the n = 75 profile so far, steps 40–44 would run on the fast path. Using step 43's measured rate (83 s for 9.1 G raw entries), the saving is roughly 300–350 s by step 44, with more on later steps of that size. This is an extrapolation from one step.
* Peak memory is bounded by the rule in §2. It will be checked against VmHWM on the real machine. It is not a statement about swap behaviour if the OS reserve of 4 GiB turns out to be too small on a shared machine.
* A step whose *output* table N does not fit in RAM is still out of scope (no RAM-only design can handle it). The spill rule makes that case fail early with a clear message instead of swapping.


## 8. Implementation (2026-11-25)

### Done as designed

* **§2.** Every step uses RAM runs and the A1 merge; `sm_should_use_ext` is consulted only with `HAM_SM_LAYOUT=ext`.
  * At each flush, a run is kept in RAM if the step's RAM runs stay within `spill_limit` = budget − input table − worker buffers − smallest merge plan. Otherwise the run is appended to the worker's spill file.
  * The budget is RAM − 4 GiB − the CRT snapshot held in RAM.
* **§3.** One spill file per worker, created with `mkstemp` in the spill directory and unlinked immediately. A1 merges RAM streams and file streams together:
  * file streams are read through a window (`pread`, then `posix_fadvise(DONTNEED)` where available);
  * splitters and slice bounds come from single-entry `pread`s;
  * the plan gives file buffers up to half the headroom (256 KiB – 8 MiB each) before sizing the release granule.

  A warning is printed if the spill directory is tmpfs. The default is `$TMPDIR`, else `/tmp`; on your Linux machine that is `/tmp`, which may be tmpfs there — the warning will say.
* **§4.** `sm_introduce` now also returns the exact `u128` sum of the counts.
  * Just before each sweep, U_s = 2^n_back · sum. The anchor snapshot is taken at the first step with U_s ≥ 2⁶⁴: in RAM if it fits `crt_snap_bytes` (default min(RAM/16, 64 GiB)), otherwise written once to the spill directory with P parallel `pwrite`s.
  * After that, a rolling snapshot is taken of every step input ≤ min(1 GiB, `crt_snap_bytes`); it replaces the anchor.
  * Snapshots are now taken after introduce, i.e. just before the sweep. The replay resumes there.
* **Options.** `count_hamiltonian_paths_sm(spill_dir=…, crt_snap_bytes=…)`, and `ham_count --spill-dir DIR --crt-snap-gb GB`.
* **Instrumentation.**
  * `[inst]   runs: total=… spilled=… (GB) resident_runs=… spill_limit=… dir=…`;
  * `[inst]   crt: log2U=… snapshot=none|anchor(ram)|anchor(disk)|rolling held=…GB`;
  * the A1 line now also shows `K_file`, `Bf` and `R_file`;
  * `n_runs` on the `step=` line is now the **total** number of runs over all workers. It was worker 0's count, and 0 under A1.

### Changes beyond the design

* **A1 replaces A0 as the automatic fallback.** Even A1's smallest plan (T = 1) peaks below A0, and A0 cannot read spilled runs. A0 remains available with `HAM_SM_MERGE=a0`. This also removed a latent failure: a step with no runs at all, reached through A0's pairwise path, tried to allocate 0 bytes and aborted.
* **The 100 % CPU phase you saw in `top`.** In the old ext merge (`sm_global_ext_merge_par`), after the parallel merge threads finish, **Phase 5 compacts the T output ranges with one serial `memmove` over the whole output**: up to ~110 GB at n = 75 step 44. Before the threads start, the T × K stream buffers are also allocated and pre-filled serially. A1 has neither: it writes private chunks and copies them into place in parallel.

  A second serial phase existed on every path: `free()` of the input table, a single `munmap` of up to ~100 GB, in which the kernel frees every page on one core. Tables ≥ 1 GiB are now released by T threads with `madvise` first (`sm_free_big`, Linux), then unmapped.
* **Test hooks** (they do not change defaults): `HAM_SM_SPILL_LIMIT=<bytes>` sets the spill limit directly; `HAM_SM_LAYOUT=ext` selects the old layout.

### Tests (pass in the cloud container; see the reply for the VM)

* `tests/test_sm_merge.py`, n = 36 and 41. Per-step digests and counts are identical across:
  * A0;
  * A1;
  * A1 with a tight budget;
  * partial spill (to a `TMPDIR` set by the test);
  * everything spilled;
  * the old ext layout;
  * nominal RAM so low that every run spills.

  It also checks that runs really were spilled where expected, and none otherwise.
* `tests/test_sm_crt.py`: exact paths and cycles under 12-, 16- and 20-bit emulated words with the anchor in RAM and on disk; residue mode; three moduli; low RAM. New: the first wrap never precedes the first step with U_s ≥ 2^b (b = 12, 16, 20). With b = 12, for example, a wrap was first possible at step 12 and first happened at step 25.
* **Larger checks** against the unmodified code (per-step digests): n = 45 and 47 identical, with no spill and with a 0.2 GB spill limit (1 246 runs spilled; peak RSS 959 MB against 1 069 MB without spilling). Each n = 47 run took 39–41 s; the original code took 66 s.

### Costs to be aware of

* The anchor snapshot is held from the first risky step to the end of the run, or until a rolling snapshot replaces it: up to `crt_snap_bytes` of RAM (≤ 47 GB on the Linux machine), and it is subtracted from every later step's budget. With `--crt-snap-gb 0` it always goes to disk instead (one parallel write).
* Rolling snapshots `memcpy` ≤ 1 GiB per small step after the anchor.
