# Sort-merge DP: 64-bit state counts at n = 74 — diagnosis and proposed fix

Status (2026-11-25): **items 1, 2 and 4 implemented**; see the last section.

## What happened

`data/linux/new_74_sm_new.txt` (kpw work ordering, new `machine.yaml`) runs all 74 steps (DP 733 s), and then `count_hamiltonian_paths_sm` raises `OverflowError`. The flag `sm_val_overflow` is set by `SM_VAL_ADD` whenever two counts of the same state wrap when added during dedup. It is only checked after the last step, so "dies at the very last step" means "the check runs then". The wrap itself may have happened at any step, but most likely near the end (below).

## Why it is (almost certainly) genuine and not caused by A0/A1

* `SMEntry.val` is a `u64`. The comment "safe for n ≤ ~110" in `ham_dp_c.c` is wrong: it reasons from the *final* count, but a state's count is the number of partial path systems on the processed vertices with that boundary, and nothing bounds it by H(n). Many of these partial systems never become Hamiltonian paths; at the last steps the closing eliminations discard most of them.
* The tail of the new run is much narrower than with the MaxSAT ordering:

  | step | states (kpw order, new run) | states (MaxSAT order, `profile_74new.txt`) |
  |---|---|---|
  | 72 | 57 | 245 |
  | 71 | 4 492 | 431 514 |

  The same mass of partial systems is concentrated in 4× fewer states, so the per-state counts are ~4× larger. With H(74) = 2.18·10¹⁸ already within a factor 8.5 of 2⁶⁴, a few-fold concentration is enough to wrap.
* The merge variants never change the values. In every test, A0 and A1 gave per-step digests identical to the old code, including the sum of all counts. The overflow test sits in the same `SM_VAL_ADD` as before.

This is an inference, not a proof. The check that settles it is below.

## What the wrapped run still knows

All the DP does to counts is add them: transitions copy a count, dedup adds, and the final step accumulates into a `u128`. So with wrapping `u64` arithmetic **every count, and the result, is exact modulo 2⁶⁴**. The run computed H(74) mod 2⁶⁴, and the Python code currently throws it away when it raises.

The previous runs give H(74) = 2 176 025 661 130 535 741 < 2⁶⁴. **If the residue of the new run equals that number**, that confirms both that the wrap is genuine and that A0/A1 computed the right thing.

## Proposed changes

1. **Report, don't discard.** On overflow, put the residue `paths mod 2^64` and the first step at which a wrap occurred into the exception, as attributes and in the message, instead of only "INCORRECT". The C side records the step with an atomic compare-and-swap on first wrap. Cost: none.
2. **Exact results without wider entries: a second run modulo a prime (CRT).**
   * When run 1 overflowed, rerun the same ordering with all counts reduced mod p = 2⁶¹ − 1 (Mersenne: `addmod` is an add, a shift and a mask). Then combine:

     X ≡ r₁ (mod 2⁶⁴),  X ≡ r₂ (mod p)  ⇒  X = r₁ + 2⁶⁴ · ((r₂ − r₁) · 2⁻⁶⁴ mod p).

     This is exact whenever X < 2⁶⁴·p ≈ 2¹²⁵.

     *That hypothesis needs a certificate.* I have no useful a-priori bound: degree-product bounds for these graphs are ~2²⁷⁰. Two options:
       * **(a)** A third modulus, which raises the limit to ≈ 2¹⁸⁶. This is still uncertified, only far more comfortable: H(75) ≈ 2⁶⁹ and the sequence grows by ~2² per n.
       * **(b)** Certify with a floating-point estimate. Summing non-negative numbers in double precision through the DP gives X̃ with |X̃ − X| ≤ N·2⁻⁵³·X, where N is the length of the longest addition chain (≤ total additions, ~10¹² here). That rigorously brackets X, e.g. X < 2¹²⁵.

       (b) would need a second value field, i.e. more memory, unless it is done in its own run, i.e. more time. I lean to (a) plus a sanity check against the growth of the known values, but this is your call.
   * Memory and per-step speed are unchanged. The price is a second run, and only for those n where the first one wrapped. In the hot path, the modulus is a runtime switch: one predictable branch in `SM_VAL_ADD`.
   * `ham_count` does this automatically (`--on-overflow crt`, default) or stops after run 1 (`--on-overflow residue`). The cycle count gets the same treatment.
3. **Not proposed: wide (`u128`) entries.**
   * Packed at 24 B they cost 1.5× memory and bandwidth on every step, and the memory peak is what limits n ≥ 75.
   * A per-step switch to wide entries "when needed" cannot be made safe cheaply. A wrap is only detectable after the merge, and the RAM path has already freed that step's input by then. A bound computed before the step (2^n_back · Σ counts) is too pessimistic in the middle of the run.
4. Fix the wrong comment on `SMEntry`.

## Unrelated, found in the same files

* `tune_params.py` writes YAML aliases (`*id001`) into `machine.yaml`, because the same lists appear under `_cacheprobe` and at top level. This is harmless but ugly; the fix is to dump with aliases disabled.
* Timing so far (not instrumented, so not attributable to individual changes). The ordering, the parameters and the code all changed at once:
  * n = 69: 410 s now vs. 1465 s (April, MaxSAT ordering);
  * n = 74: 733 s for the DP vs. 945 s (best April run).


## Implementation (2026-11-25)

### The certificate: the Bregman–Minc bound we already have

`hamiltonian/bounds/bregman.py` holds the rigorous bound worked out earlier. Add a vertex joined to everything; each directed Hamiltonian path then gives a cycle cover, so

  2 H(n) ≤ per(A′) ≤ ∏_v ((d_v+1)!)^{1/(d_v+1)} · (n!)^{1/n}

by Brégman (1973). Hamiltonian cycles are covered too: deleting an edge maps (cycle, edge) injectively to a path, so C ≤ H/n ≤ H.

As an approximation of H(n) the bound is loose: 2^114.1 against H(74) ≈ 2^60.9. For the CRT, though, it is plenty:

| n | log₂ bound | log₂ H(n) |
|---|---|---|
| 69 | 104.3 | 52.6 |
| 74 | 114.1 | 60.9 |
| 75 | 116.3 | 69.1 |
| 80 | 127.1 | — |
| 90 | 147.0 | — |
| 100 | 168.5 | — |

One prime (2⁶¹−1) together with the 2⁶⁴ run certifies results up to n = 79 (2^125 > 2^(bound+1)), two primes to about n = 104, and three beyond that. `count_hamiltonian_paths_sm` computes the bound from the graph's own degrees. It uses the smallest number of moduli k with 64 + Σ log₂ p_i ≥ bound + 1 (a one-bit margin covers the floating-point evaluation of `lgamma`). It asserts that the CRT result lies below the bound.

### What runs when a count wraps (`on_overflow="crt"`, the default)

* **Main run.** Arithmetic is mod 2⁶⁴ (wrapping). The first wrap is recorded together with its step (`sm_overflow_step`).
* **Snapshot.** At the start of each step whose input table is at most `crt_snap_bytes`, and until the first wrap, the DP copies that table and the frontier state. The default budget is min(1 GiB, RAM/64). Any snapshot taken at or before the wrap step has exact counts.
* **Replay (the cheap case — yours).** After the main run, the steps from the last snapshot onward are replayed once per prime, with counts mod p (`addmod`, values < 2⁶²). At n = 74 the wrap is in the last few steps, where the tables have 10⁴–10⁷ states, so the replays should take seconds rather than another full run. This is an estimate: I don't know the exact wrap step for your run.
* **Fallback.** If no snapshot precedes the wrap (a wrap at a big step), the whole DP is rerun mod each prime.
* **Combination.** Garner's CRT with moduli 2⁶⁴, p₁, …. The result is certified by the bound. One line on stderr says what was done, for example: `# sort-merge: counts wrapped mod 2^64 (first at step 72); exact value by CRT with 1 prime(s) < 2^62 (replayed steps 72..73); Bregman bound 2^114.1 < modulus 2^125.0`.

`on_overflow="residue"` raises `SMOverflow`, carrying `.residue` (paths mod 2⁶⁴), `.cycles_residue` and `.step`. In `ham_count` the choice is `--on-overflow {crt,residue}`.

### Other changes

* The overflow flag was never reset, so in a multi-n `ham_count` run every n after the first overflow would have been reported as overflowed. It is now reset at the start of each call.
* The step loop of `count_ham_paths_sm` was moved into `sm_run_steps` (restartable). Per-step state tables are bit-identical to before: digests compared at n = 45 and 47.
* `SM_VAL_ADD` gained one predictable branch (`sm_mod`) plus a test-only one (`sm_wbits`).
* The comment "u64 safe for n ≤ ~110" has been replaced by the explanation above.
* `tune_params.py` now writes `machine.yaml` without YAML aliases.
* Memory cost: the snapshot, at most `crt_snap_bytes`. Time cost: one `memcpy` of it per small step.

### Tests (`tests/test_sm_crt.py`; all pass, also on the aarch64 Linux VM)

A test hook `_word_bits=b` emulates b-bit counts (b = 12, 16, 20) at n = 41, so that wraps occur from step 25 or 35 on. The tests check:
* exact paths and cycles, by both replay and full reruns;
* that the residue raised in `residue` mode is correct;
* three small moduli (patched in) combining correctly;
* the ext path;
* that the bound matches `bregman.py` and exceeds the known H(74).
