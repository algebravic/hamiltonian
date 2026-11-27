# Enumerating only the back-edge subsets that can be valid

Status: implemented 2026-11-27 (`ham_dp_c.c`, key16 only; `ham_dp_c_legacy15.c` is unchanged).

## Observation

At step t the DP introduces v = v_t, which has n_back back-edges to vertices already on the frontier. For every input state, the sweep enumerated all 2^{n_back} subsets S of these edges and applied the edges of S to v in increasing index order.

**Claim (proved): only subsets with |S| ≤ 2 can yield an output.**

*Proof.* v enters with slot value 0 (degree 0). Follow the edges of S at v, in order, through `apply_edge`:

- **First edge (v, w).** If w is interior, the edge is rejected. Otherwise v becomes a chain end (slot value > 0).
- **Second edge (v, w′).** v now has label L > 0.
  - If w′ has slot value 0, then v becomes interior (−1).
  - If w′ has a label L′ ≠ L, then v and w′ become interior and the chains merge. At the last step, a merge that completes the path gives `KEY_CLOSED` instead.
  - If L′ = L, the edge is rejected (premature cycle). At the last step with nothing left in S, it is counted as a Hamiltonian cycle and then also rejected for the path count.
- **Third edge.** After two accepted edges, v is interior or the key is `KEY_CLOSED`, and `apply_edge` returns 0 for both. So every subset with |S| ≥ 3 is rejected.

The cycle count is unaffected too. A cycle is counted only at an edge j with no further edges in S (`!(S >> (j+1))`), so only at the second edge of a 2-subset. The subsets of size ≥ 3 never contributed. □

So of the 2^{n_back} subsets, at most 1 + n_back + C(n_back, 2) can produce an output: 2, 4, 7, 11, 16, 22, 29, 37 for n_back = 1–8, against 2, 4, 8, 16, 32, 64, 128, 256.

## Change

`small_subsets(nb, masks)` lists these masks: the empty set, then the singletons, then the pairs. The subset loops now run over this list:
- `sm_worker_runs`, the sort-merge DP;
- `fused_sweep`, the EH backend;
- `sm_sample_outputs`, the worker-cap sampler.

Within a subset the edges are still applied in increasing index order. So every output key, every count and every cycle count is unchanged; only the order in which the outputs are generated differs, and sorting and merging do not depend on that order.

`MAX_NBACK` = 16 still bounds n_back. The mask list has at most 137 entries.

## Verification (Linux VM)

- **Per-step checksums identical, before and after the change** (`HAM_SM_CKSUM`, order-independent digest of each step's table):
  - G_41, BFS ordering: `--prune drop` and `--prune none`;
  - G_50, kpw ordering: both modes;
  - G_36, BFS ordering: `--prune none`.
- **Totals identical:** path counts and cycle counts, sort-merge and EH backends, in all five cases.
- **Tests pass:**
  - `test_dp_correctness.py` (random graphs with many back-edges and width-16 graphs against Held–Karp; G_n widths);
  - `test_sm_merge.py`, `test_sm_crt.py`, `test_worker_cap.py`, `test_kpw.py`.

## Effect

- **G_58** (n_back ≤ 4), VM, 4 threads, same ordering and buffer size: state generation took 94.8 s before and 83.4 s after (−12%). The DP time went from 131.7 s to 128.9 s; these are single runs, within noise.
- **Graphs with many back-edges:** `test_many_back_edges[2]` (random graphs with a hub vertex of about 9 back-edges, two pruning modes) previously ran past the 3-minute limit of the VM. It now takes 39 s. `test_many_back_edges[1]` together with the width-16 test fell from 101 s to 36 s.

## Consequence for the cost model

The generation cost per input state is now proportional to the number of masks, 1 + n_back + C(n_back, 2), not to 2^{n_back}. The work model's per-step multiplier e_t = 2^{nb_t} therefore no longer describes the code. See the cost-form section of `doc/work_model_redesign.md`.
