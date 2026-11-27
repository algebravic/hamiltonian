# Choosing the worker buffer size per step

Status: implemented 2026-11-27, with one change from the design: the output is predicted by **sampling**, not from history (see "Implementation" at the end).

## Problem

Each of the P workers of a sort-merge step has two buffers of `SM_WORKER_CAP` entries of 16 bytes: an output buffer and a radix-sort scratch buffer. They are allocated at the start of every step (`sm_fused_sweep`, `ham_dp_c.c` ≈ line 3830) and freed before the merge. `SM_WORKER_CAP` is one fixed number for the whole run, taken from `machine.yaml`.

The buffers take memory away from the runs. The in-sweep spill limit is

  spill_limit = B − F_in − 2P·C·16 − M_res,

where:
- B is the merge budget (RAM − 4 GiB − held CRT snapshot);
- F_in is the input table;
- C is the buffer size;
- M_res is the merge reserve.

**Example: n = 63 on the M3 Pro (18 GB, P = 6, C = 32 M).** The buffers take 6.4 GB.

- **What happened.** At step 41 the input was 166 M states and the workers produced 431 M contributions. That is about 6.4 GB of runs after the deduplication within each run, which removed only about 1.07×. The spill limit was 5.3 GB, so 1.2 GB went to disk. Steps 39–42 all spilled about 1 GB each.
- **The alternative.** With C = 16 M the buffers would take 3.2 GB, the limit would be about 8.4 GB, and nothing would spill.
- **The waste at the other extreme.** At n = 58 in the VM, no worker ever produced more than about 3.3 M contributions in a step, so most of each 32 M buffer was never used.

## What the buffer size costs

n = 58, Linux VM, 4 threads, same build and ordering; one run each:

| C | DP time | generate | flush (radix sort) | merge | max runs per step | in-run dedup |
|---|---|---|---|---|---|---|
| 32 M (512 MB) | 131.7 s | 94.8 s | 19.9 s | 18.1 s | 4 | up to 1.74× |
| 1 M (16 MB) | 125.7 s | 85.1 s | 14.8 s | 25.9 s | 24 | up to 1.16× |

**Caveat, found during implementation:** the VM has only 3 GB of RAM, so its merge budget is 0 and every run in both of these measurements was written to the spill directory. The comparison is therefore between two all-spill runs. It still shows the flush/merge trade-off, but it is not a measurement of the in-RAM case.

Within the noise of single runs, a 32-fold smaller buffer cost nothing in time:
- **flushes got cheaper**, because the buffer is nearer to cache;
- **merges got dearer**, because there were 6× as many streams;
- **fewer duplicates were removed within each run**, because each run is shorter.

The last point matters for memory: shrinking C frees buffer memory but makes the runs somewhat larger.

**C never affects correctness.** Any C ≥ 1 gives the same tables; only time and memory change. The per-step checksum test (`HAM_SM_CKSUM`) will check this.

## Rule

Before the workers start, for step t with n_back = b:

1. **Predict the raw output R̂.**
   - Let N_in be the input size. Use R̂ = N_in · g_b, where g_b is the largest ratio (contributions written / input size) observed at earlier steps of this run with the same n_back.
   - With no history, use g_b = 2^b. This is a true upper bound: every state yields at most 2^b contributions.
   - The ratio is measured after pruning, so it includes the pruning's effect.
2. **Bound the useful size.** A buffer larger than one worker's share of the output is never filled, so set C_need = ⌈R̂ / P⌉.
3. **Set the limits.**
   - C_max = `SM_WORKER_CAP` from the yaml. Its meaning changes from "the size" to "the largest size".
   - C_lo = max(`SM_WORKER_CAP_MIN`, ⌈R̂ / `SM_MAX_RUNS`⌉). The defaults are a minimum of 1 M entries (16 MB) and at most 1024 runs per step, so that the merge's stream count stays bounded.
4. **Memory test.**
   - fits(C) ⇔ F_in + 2P·C·16 + R̂·16 + M_res ≤ B.
   - This assumes no deduplication within runs, i.e. runs take the full R̂·16 bytes. That is conservative, and at the sizes where spilling matters the observed in-run dedup was only 1.07–1.12×.
5. **Choose C.**
   - Start from C = min(C_max, C_need), rounded up to a power of two.
   - If fits(C), use it.
   - Otherwise halve C while C > C_lo and not fits(C), and use the first C that fits.
   - If even C_lo does not fit, spilling is unavoidable: use C_lo. The smallest buffers give the largest spill limit, and hence the smallest spill.
6. Compute the spill limit from the chosen C, with the existing formula, and run the step.

**Prediction errors are harmless for correctness.** If R exceeds R̂, the existing in-sweep spill rule writes the excess runs to disk, exactly as now. A prediction that is too low costs only a spill that a smaller C might have avoided. A prediction that is too high only shrinks the buffers unnecessarily.

**Check against the example.** At n = 63, step 41: R̂ ≈ 431 M, so C_need = 72 M and C_max = 32 M. fits(32 M) fails (6.9 GB of predicted runs > 5.3 GB), but fits(16 M) holds (6.9 GB < 8.4 GB). The rule picks 16 M, with no spill predicted. This is a prediction; the run has not been repeated.

## Option for later: shrinking within a step

Workers could flush early, with a smaller effective C, when the resident runs approach the spill limit in the middle of a step. That would correct bad predictions as they happen, but it touches the hot loop and the spill bookkeeping. It is not part of this round.

## Configuration and reporting

- **yaml:** `SM_WORKER_CAP` is now the maximum. New keys `SM_WORKER_CAP_MIN` (default 1048576) and `SM_MAX_RUNS` (default 1024).
- **`ham_count --worker-cap {auto,fixed}`** (default `auto`). `fixed` restores today's behaviour, for A/B comparisons.
- **The log header** prints the policy and the limits.
- **`--instrument`** prints one line per step:
  `[inst]   cap: C=16M (need=72M max=32M lo=1M) pred_raw=431M actual=431M g=2.60 fits=yes`
- **The stale `[ext]` tag in the `-v` step lines** (it still applied the old ext predicate) is replaced by the layout actually used: `[sm]`, or `[sm+spill k/K]` when k of the step's K runs spilled.

## Where

`ham_dp_c.c` (key16 only; `legacy15` is unchanged):
- `sm_fused_sweep` chooses C before the spill-limit block and uses it for the allocations and in the spill limit.
- `SMWorkerState` gains `cap`. The flush test `ws->buf_len == sm_cfg.worker_cap` (≈ line 2585) becomes `ws->buf_len == ws->cap`.
- `sm_run_steps` keeps g_b per n_back and updates it after each sweep from the workers' written-entry counts.
- A new setter `sm_set_cap_policy(int automatic, size_t cap_min, size_t max_runs)`.
- The ext layout (`HAM_SM_LAYOUT=ext`) keeps the fixed C, since its capacity calculations assume it.

`ham_dp_c.py` declares and calls the setter, reads the new yaml keys, and prints the header line. `ham_count.py` adds `--worker-cap`.

## Tests

1. **Identical tables.** The per-step checksums are identical for `auto` and `fixed`, and for C_max ∈ {64 K, 1 M, 32 M}, on G_36 and G_41 (as `test_sm_merge.py` does).
2. **Bounds.** The chosen C always lies in [C_lo, C_max]. If C_need < C_max, then C ≤ the next power of two above C_need.
3. **Avoiding a spill.** With the test budget hook `HAM_SM_MERGE_BUDGET` set so that `fixed` spills at some step of G_41, `auto` does not spill, and the count is unchanged.
4. **Timing (on your machines).** n = 61–63 on the M3 Pro with `auto` and with `fixed`. Expected: no spills at n = 63, and similar times at n ≤ 62. This is a hypothesis to measure, not a promise.

## Implementation (2026-11-27)

**Prediction by sampling, not history.** Before the workers start, `sm_sample_outputs` runs the step's transitions, eliminations and pruning on up to 65 536 evenly spaced input states, single-threaded, and scales the number of written outputs by N_in / sample size. Because the input table is in hash order, the sample is in effect random.

This replaces the per-n_back history ratio of the design. History has no data at the first step with a new n_back, which is exactly where the 2^b fallback would be far too high. The sample costs at most 65 536 × 2^b transitions per step, a few milliseconds.

At n = 57 the prediction matched the actual output at every step to the reported precision (`pred=4M actual=4M`). `sm_choose_cap` adds a 25% margin to C_need.

**Code:**
- `ham_dp_c.c`:
  - `sm_set_cap_policy`, `sm_sample_outputs`, `sm_choose_cap`;
  - `SMWorkerState.cap`, used for the allocations and the flush test;
  - the spill limit now uses the chosen C;
  - the `-v` step tag shows the layout actually used (`[sm]`, `[sm+spill k/K]`, `[ext]`);
  - `--instrument` prints one `cap:` line per step.
- `ham_dp_c.py`:
  - `worker_cap="auto"|"fixed"`;
  - yaml keys `SM_WORKER_CAP_MIN` and `SM_MAX_RUNS`;
  - the header prints both.
- `ham_count.py`: `--worker-cap {auto,fixed}`.
- The ext layout keeps the fixed C.

**Checks (VM, nominal RAM 16 GB):**
- Counts for n = 30–50 are identical under `auto` and `fixed`.
- `tests/test_worker_cap.py`:
  - per-step checksums are identical across (C_max, policy, C_min) ∈ {(32 M, fixed, 1 M), (32 M, auto, 1 M), (1 M, auto, 1 K), (64 K, fixed, 1 K), (4 K, auto, 256)} on G_36 and G_41;
  - with a 300 MB budget, `fixed` spills and `auto` does not.
- n = 57 with a 600 MB budget gives the same count. `fixed` spilled at every step; `auto` spilled at none.
- `test_sm_merge.py`, `test_sm_crt.py`, `test_build_cache.py`, and two of the correctness tests (random width-16 graphs and back-edges[1]) all pass.

**Not yet measured:** times on the M3 Pro at n = 61–63 with `auto` against `fixed`.
