# Two counting bugs found while investigating the n = 75 crash, and a proposal

Status (2026-11-26): **both bugs fixed; the 16-slot encoding of §4 is implemented** (§5). The old encoding is kept as `ham_dp_c_legacy15.c`.

## 1. The crash after step 57

* `data/linux/new_75_sm_new.txt` ends right after step 57 prints `introduce: si=9311M`, i.e. inside step 58. Its input table is 9.31 G states (149 GB).
* There is no Python traceback and no C message. That pattern is what the Linux OOM killer produces (SIGKILL), but the log cannot prove it. `dmesg -T | grep -iE "killed process|out of memory"` on the machine would.
* The run used the binary compiled at its start (before the spill rule), which still decided "ext" vs RAM before the sweep.
* At step 57 the measured merge peak was already 262 GB. The sweep of step 58 needs the 149 GB input plus 69 GB of worker buffers plus the runs, which is plausibly more than the 755 GB available.

This is moot, though, because of §3: that run's count would have been wrong.

## 2. Bug 1: back-edges beyond 8 silently dropped (fixed)

`widxs[8]` together with `n_back < 8` in both the EH and the sort-merge step loops dropped the 9th and later back-edges of a vertex without any message.

* **Found by:** an independent Held-Karp count (`tests/test_dp_correctness.py`) on random graphs. For example, one 21-vertex graph whose last vertex has 9 back-edges: exact 2, the DP gave 1.
* **Fix:** `MAX_NBACK` = 16, with a hard error beyond that.
* **Impact on your results: none, as far as the logs show.** I scanned every `profile_*` log in `data/`: no run ever reached n_back = 8, the value at which the cap could act. The maximum was 7, in `profile_square_diff_30*`; for G_n the maximum is 5–6.

## 3. Bug 2: 16 frontier slots overlap the flag bits (guarded)

The state key packs 4 bits per frontier slot plus 2 flags (NC = bit 62, MARKER = bit 63), so it holds **15 slots**. The guard was `fs > 15` *before* introduce, which lets fs = 15 through. The vertex being introduced then becomes slot 16, at bits 60–63, overlapping the flags.

* The frontier inside a step is (vertex separation of the ordering) + 1. That is `max_fw`/`pw` + 1 as printed by `ham_count`, whose `frontier_width` counts *after* elimination.
* **So any ordering with pw = 15 needs 16 slots.**

Evidence (Held-Karp and G_n itself):

| test | in-step width ≤ 15 | in-step width 16 |
|---|---|---|
| random graphs n = 18–21 vs Held-Karp (after fixing bug 1) | 45 / 45 correct | 7 of 15 wrong |
| G_36, G_38, G_40, perturbed BFS orders vs the known counts | 12 / 12 correct | 9 / 9 wrong (e.g. G_36: 1728, 105, 61 instead of 12 960) |

**Consequences**

* All counts from orderings with pw ≤ 14 are unaffected. That covers n ≤ 74, including both n = 74 runs, which agree.
* **The April n = 75 value 652 211 775 281 872 067 084 came from a pw = 15 ordering, so by this evidence it is not reliable.** The same holds for the kpw n = 75 run, which also has pw = 15. The places where the n = 75 value appears with a caveat (`talk2.tex`, the papers) should now treat it as withdrawn.

**Guard (committed).** `count_hamiltonian_paths_sm/_c/_peh` now refuse such orderings up front, via `in_step_width()`, with a clear message. The C loops check `fs + 1 > 15` as a second line of defence. Tests: `tests/test_dp_correctness.py` (width 16 refused, width 15 exact, ≥ 9 back-edges exact). Before the fix the old code was wrong on all four back-edge test graphs.

## 4. Proposal: 16 slots in the same 64-bit key (enables pw = 15, hence n = 75)

**Claim (to be checked in review).** The NC flag is only ever needed inside the last step. A chain whose two endpoints have both left the frontier ("closed") is a whole component. If that happens at a step s < n − 1, vertices remain to be introduced, and none of them can join it: the frontier entries of a closed state are all interior (−1), and any later segment would be a second component. Such a state can therefore never be counted, and is discarded today only later, at the next elimination or at `new_nc > 1`. Hence:

* at step s < n − 1, a closure discards the state immediately, so no NC bit is needed;
* at step n − 1, a closure with all other entries interior adds the count to the total immediately, provided no further edge of the current subset is taken. Such an edge would touch an interior vertex and is invalid anyway.

The MARKER bit only makes the initial key non-zero, because 0 is the "invalid" return of `apply_edge`. It can be replaced by the sentinel `UINT64_MAX`. That value is never a valid key: it would need all 16 slots equal to label 13, and a label occurs at most twice.

Without both flags, 16 slots × 4 bits = 64 bits. **Memory and speed are unchanged**, and vertex separation 15 becomes possible. Pathwidth 16, likely at some n > 75, would still need a 128-bit key; that is a separate, larger change, at 1.5–2× memory.

**Validation plan**
* Held-Karp at in-step width 16, on random graphs and on G_n with perturbed orders.
* The n ≤ 74 results reproduced exactly, both counts and per-step digests where the flags never mattered.
* The cycle count (which uses NC at the last step) against Held-Karp's cycle count.

You approved this on 2026-11-26, keeping the old semantics in a separate file.


## 5. Implementation (2026-11-26)

### Files

* **`hamiltonian/claude/ham_dp_c.c`**: the new key. It has 16 slots of 4 bits, `MAX_FS_FAST` = 16, and no `KEY_NC_BIT`/`KEY_MARKER`.
  * A closure before the last step returns "invalid" at once.
  * At the last step, a closure by elimination adds the count at once. A closure by an edge yields the sentinel `KEY_CLOSED` (`UINT64_MAX`), which rejects any later edge of the same subset and is counted by `apply_elim_seq`.
  * The cycle check skips `KEY_CLOSED`.
  * In the EH path's non-fused sweeps (B), a closure at the last step is counted directly, and the final-step lookup of the closed key is gone.
  * The initial key is 0, the empty frontier, which is never an output.
  * `eliminate_slot` avoids the shift by 64 when slot 15 of 16 is removed.
  * The checkpoint format version is 3, so old EH checkpoints, which hold flagged keys, are rejected rather than misread.
* **`hamiltonian/claude/ham_dp_c_legacy15.c`**: a frozen copy with the original semantics. It still has every other improvement (A1, the spill rule, the CRT, the fixed back-edge limit and the 15-slot guard).
* **`ham_dp_c.py`**: select the variant with `set_c_variant("key16" | "legacy15")` or `HAM_DP_VARIANT=legacy15`. `MAX_STATE_SLOTS` and the width check follow the variant.

### Verification

| check | result |
|---|---|
| Held-Karp, random graphs of in-step width 14, 15, 16 (45 graphs + 6), **paths and cycles**, sort-merge and EH | all exact |
| G_36, G_40, G_44 with BFS orders: key16 vs legacy15, paths and cycles, SM and EH | identical (e.g. G_44: 5 301 127 paths, 10 091 cycles) |
| G_36/40/44 with perturbed orders of in-step width **16** (8 orders) | all exact with key16 (legacy15 refuses them) |
| n = 47 vs the original code | same count (68 361 609); 48 s vs 66 s |
| `tests/test_sm_merge.py`, `test_sm_crt.py` | pass with both variants |
| `tests/test_dp_correctness.py` (new cases: G_36 widths 15/16/17 for both variants, random width-16 graphs vs Held-Karp) | pass |

Per-step digests are *not* comparable between the variants, by design: the keys differ (no marker bit), and legacy15 keeps the doomed closed states for a while before they die. The final counts are what must agree, and they do.

### What this means for n = 75

A pw = 15 ordering, i.e. in-step width 16 (which both n = 75 runs used), now runs correctly under key16. Pathwidth 16 (in-step width 17) is refused with a clear message, and would need a 128-bit key.
