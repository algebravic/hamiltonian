# Completability pruning for the sort-merge DP (Du's idea)

Status: on by default since 2026-11-26. It is chosen with `ham_count --prune {drop,report,none}` (`count_hamiltonian_paths_sm(..., prune=...)`), and the default is `drop`. It was first tested as the environment variable `HAM_SM_PRUNE` (0/1/2 = none/report/drop), which is no longer read. It lives only in `ham_dp_c.c` (key16); `ham_dp_c_legacy15.c` has no pruning.

## Idea

Zhao Hui Du's `path4.cpp` discards a partial state when the vertices not yet finished can no longer form a single path with the existing fragments. It does this in two ways:
- **Connectivity:** the remaining graph must be connected.
- **Degree:** a vertex that can no longer reach degree 2 is pruned.

Our DP has no such test. A dead state is carried, multiplied by up to 2^n_back each step, sorted and merged, until it dies at an elimination or at the final step. Even the simplest case is missed: three or more chains whose other end has already left the frontier can never join into one path, yet such states survive to the end.

## Setting

After step s (introduce v = order[s], add the back-edges, eliminate), the frontier is F′ and the future set is U = {order[s+1], …, order[n−1]}.

**Edges already decided.** Every edge with both ends in F′ ∪ (eliminated vertices) has been decided. The only edges still undecided are those with an end in U.

**Slot states.** Each slot k ∈ F′ holds one of:
- 0: degree 0;
- −1: interior, degree 2;
- a label L > 0, meaning degree 1, the end of a chain. Each label occurs at most twice. A label that occurs **once** marks a chain whose other end has been eliminated with degree 1. That other end is therefore one of the two endpoints of the final path; call this a *dangling* chain.

**Future degrees.** Every slot in F′ has at least one neighbour in U, since otherwise it would have been eliminated. Write fdeg(k) for the number of its neighbours in U.

## Necessary conditions for a state to be completable (for s < n − 1)

Let N = {k ∈ F′ : slot k ≠ −1}, the non-interior slots. Let H be the graph on N ∪ U with:
- all edges of G inside U;
- all edges of G between N and U;
- one "chain link" between the two slots of each label that occurs twice.

Any completion adds a set of edges inside U and between N and U. Together with the chains, those edges form one path that covers N ∪ U. Hence:

**(C) Connectivity.** H is connected, and N is non-empty (U is non-empty because s < n − 1).

**(D) Endpoint budget.** The final path has exactly two endpoints. Each of the following is forced to be an endpoint, and they are distinct vertices:
- the dangling end of each label that occurs once (an eliminated vertex);
- each slot k with value 0 and fdeg(k) = 1;
- each w ∈ U whose H-degree is 1.

The H-degree of w is its degree inside U plus the number of its neighbours in N. If any w ∈ U has H-degree 0, the state is dead. Otherwise the state is dead if the number of forced endpoints exceeds 2.

Both conditions are necessary, not sufficient, so pruning never removes a completable state. The count is therefore unchanged. This is the claim to verify.

## Implementation

**Per step,** computed once in `sm_run_steps` before the sweep, when s < n − 1, into a global `sm_prune` that the workers read:
- the post-elimination frontier list (the slot order of the output key);
- the components of G[U], by union-find. For each slot k: `cmask[k]`, the bitset of components containing a neighbour of k, and `fdeg[k]`. If there are more than 64 components, (C) is skipped for that step;
- the list of w ∈ U with at least one neighbour in F′, each with `udeg(w)` (its degree inside U) and `amask(w)` (the bitset of its adjacent slots);
- `e_fixed`: the number of w ∈ U with no frontier neighbour and udeg(w) = 1. If some such w has udeg 0, every state is dead.

**Per output key** (in `sm_worker_runs`, after `apply_elim_seq`):
1. Decode the slots into masks: `nonint`, `zero`, and the partner masks by label. Count the labels that occur once (e₀).
2. Check (D) using popcount on `amask(w) & nonint`.
3. Check (C) by a fix-point over bitsets. Start from one slot of `nonint`. Alternately add the components touched by the reached slots, the slots touching the reached components, and the label partners. The state passes if every slot of `nonint` and every component is reached.

The cost is O(fs × iterations + |{w}|) per output key.

**Modes:**
- `report` (formerly `HAM_SM_PRUNE=1`) counts only. After the merge, every deduplicated state is classified (live, dead by D, or dead by C) and one line per step is printed: `[prune] step=… states=… deadD=… deadC=…`. The states are kept, so the run is identical to an unpruned run apart from the time spent checking.
- `drop` (formerly `HAM_SM_PRUNE=2`; the default) drops dead keys inside the worker, before they reach the buffers, and prints `[prune] step=… dropped_raw=…`.

## How it will be evaluated

1. **Correctness.** With `HAM_SM_PRUNE=2`, the count must be unchanged for every n in a range, checked against `data/b071983.txt`, and the existing test suite must pass.
2. **Potential.** Mode 1 at n = 50 and n = 58 measures the fraction of states that are dead at each step.
3. **Benefit.** Compare mode 2 against mode 0 at n = 58: states per step, peak states, and wall time.

## What is not included

Du's forced-edge step commits an edge early when an endpoint has only one option. It does not fit our scheme, where an edge is decided when its later end is introduced. It could be emulated by pruning: when a slot with a label has exactly one remaining future neighbour, that edge is forced. That is a tightening of (D); it is left for later, depending on what the measurements show.

## Results (2026-11-26, Linux VM, 4 threads, gcc 11.4, `--ordering kpw --sort-merge`)

These are single runs each. The VM's orderings differ from the Mac's, because the DAG cache and the kpw binary differ.

### Correctness

With `HAM_SM_PRUNE=2`:
- **Paths.** The count of Hamiltonian paths agrees with the unpruned run and with `data/b071983.txt` for every n from 15 to 50, and also for n = 53, 55, 56, 57 and 58. n = 58 gives 123092214818.
- **Cycles.** With `--cycles`, the path and cycle counts are identical to the unpruned run for n = 32…48, and the cycles agree with Dobbelaere's A071984 values.
- **Tests.** `test_dp_correctness.py` passes in full: back-edges, G_n widths 15/16/17, and random width-16 graphs checked against Held–Karp. So do `test_sm_crt.py` (except one test) and `test_sm_merge[41]`.
  - Two tests fail with pruning on, and both are calibration effects. `test_no_wrap_before_risky_step` expects a wrap at 12 bits, and pruned counts are smaller, so no wrap occurs. `test_merge_variants_identical[36]` expects spills at a fixed spill limit, and the pruned runs are too small to spill.
  - Both pass with pruning off, which is the default. Pruning is a necessary-condition filter, so these two tests should be run with it off.

### Potential (mode 1)

At n = 40, 25–75% of each step's states are dead. At n = 52, summed over all steps, 54.6% of the states are dead.

**Every dead state fails (D); none fails only (C).** Connectivity adds nothing on top of the endpoint budget for these graphs and orderings.

### Benefit (mode 2 against mode 0)

| n | dp time, unpruned | dp time, pruned | speed-up | peak states, unpruned → pruned |
|---|---|---|---|---|
| 55 | 9.4 s | 5.2 s | 1.8× | |
| 56 | 16.6 s | 11.2 s | 1.5× | |
| 57 | 68.2 s | 29.4 s | 2.3× | 4.47 M → 2.03 M |
| 58 | > 174 s (step 37 at 158.0 s) | 130.1 s (step 37 at 64.6 s) | 2.4× to step 37 | 24.5 M → 7.5 M (step 37) |

In the steps that dominate the run, pruning leaves 0.3–0.5 of the states, and that factor compounds from step to step.

### Conclusions and next steps

1. The gain comes from the endpoint budget (D), not from connectivity. Two things remain to be measured:
   - which part of D fires: more than two dangling chains, which is a trivial test on the key, or the forced endpoints from fdeg and H-degree;
   - whether dropping (C) makes the check cheaper without losing anything.
2. Because the test is cheap and sound, it could become the default once it has been run on the Mac and Linux machines at larger n, for example n = 69 and 74. Making it the default changes the CRT wrap points and spill behaviour, so the two calibrated tests would need `HAM_SM_PRUNE=0`.
3. The test could be applied **before** the subset enumeration, to prune input states early, and inside `apply_edge`. Both would save work in the sweep, not only in the sort and merge.

### Update (2026-11-26)

- `--prune drop` is now the default, and the two calibrated tests pass `prune="none"`.
- On the M3 iMac, n = 58 runs in 49.5 s with `drop` (`count_58_sm_pruned3.txt`). With `report` it takes 154 s (`count_58_sm_pruned2.txt`), because `report` keeps every state and adds a serial classification pass. A clean `none` run with the same ordering is still to be done.
