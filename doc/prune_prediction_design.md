# Predicting the pruning gain: design of the study

Status: design note for an experimental study. The code is `hamiltonian/claude/prune_study.py`, a Python reference DP plus drivers, and the results go in `data/prune_study/`. Nothing in the production DP changes.

## Question

For a graph G and a vertex ordering π, how much smaller is the pruned DP than the unpruned one, and can that be predicted, even within a small factor, from G and π? A secondary question: how much does the choice among the pathwidth-optimal orderings matter?

## Definitions

Fix G and π = (v₀, …, v_{n−1}). After step t, a *state* is a canonical frontier configuration, as in the DP. Three sets of states are nested:

- **R_t**, the reachable states: the table of the DP **without** pruning.
- **P_t**, the pruned states: the table of the DP **with** pruning. A state is in P_t if it can be reached by transitions all of whose intermediate states passed the tests (C) and (D) of `doc/sm_prune_design.md`, and it passes them itself. P_t ⊆ R_t.
- **L_t**, the live states: the states of R_t that lie on at least one Hamiltonian path, i.e. that have a sequence of transitions leading to a completion at the last step. L_t ⊆ P_t, because (C) and (D) are necessary for completability, and every ancestor of a live state is live.

The quantities studied are:

- the gain ρ = Σ_t |R_t| / Σ_t |P_t|, the measured quantity;
- the gap γ = Σ_t |P_t| / Σ_t |L_t|. This is how much any further necessary test could still remove; the ceiling of pruning is Σ|R|/Σ|L|.
- the per-step ratios |R_t|/|P_t| and |P_t|/|L_t|.

Work is proportional to the number of generated contributions, not to the number of states. The study records both: Σ_t |table_{t−1}| · (number of valid subsets).

## Experiment 1: exact live sets (small n)

A Python reference implementation of the DP. It has the same semantics as `ham_dp_c.c` (key16): the same transition rules for edges, eliminations and the closure at the last step, the same canonical labelling, and the same (C) and (D) tests.

1. **Forward pass without pruning.** Record R_t, the counts, and the transition multimap from each state at t−1 to its successors at t. Mark the states with a completion at the last step.
2. **Backward pass.** L_{n−2} is the set of states with a completing transition; L_{t−1} is the set of states with a successor in L_t.
3. **Forward pass with pruning,** which gives P_t.
4. **Checks:**
   - the totals equal A071983(n);
   - |R_t| and |P_t| equal the `states_out` reported by the C DP with `--prune none` and `--prune drop`, step by step, for the same ordering;
   - L_t ⊆ P_t ⊆ R_t at every step.

Range: the kpw work-optimal ordering of G_n, for n ≤ 48 or as far as the memory of a Python implementation allows.

### Decomposition of dead states

For each state σ ∈ R_t, record the three terms of the endpoint budget:

- s(σ): the number of dangling chains, i.e. labels that occur once;
- z(σ): the number of untouched frontier vertices with exactly one neighbour still to come;
- w(σ): the number of future vertices whose H_σ-degree is 1.

A state violates (D) iff s + z + w > 2, or some future vertex has H_σ-degree 0. The study reports:

- the joint distribution of (s, z, w) over R_t;
- the fraction of the dead states (R_t \ P_t, rejected at step t) attributable to each term;
- a test of the **independence predictor**. It predicts the dead fraction at step t from the marginal distributions of s, z and w alone, as P(s + z + w > 2) under independence, and compares this with the true fraction. If it is accurate, the next step is to predict the marginals from (G, π). The z and w terms depend only on which frontier slots are 0 or −1, and the s term only on the number of dangling chains.

### Pairing-only bound

For reference: if the m degree-1 frontier vertices of a step were paired uniformly among all partial matchings, the pairing term alone would give a ratio of I(m) / #{matchings with ≤ 2 unmatched}:

| m | 12 | 13 | 14 | 15 |
|---|---|---|---|---|
| ratio | 1.93 | 4.21 | 2.21 | 5.11 |

The measured |R_t|/|P_t| = 4.9 at the largest step of n = 61 (pathwidth 14) shows that the uniform-pairing picture alone does not account for it.

## Experiment 2: spread over pathwidth-optimal orderings

For each n ∈ {50, 52, 54, 55, 56}, take the orderings of `ham_kpw.kpw_candidates`:

- the optima under the work, v7 and v2 costs;
- the 3 best orderings under v2;
- 8 uniform samples from the kpw DAG, with a fixed seed.

All of them have optimal pathwidth. For each ordering, run the C DP with `--prune none` and with `--prune drop`. Record the per-step `states_out`, Σ states, the peak, and the DP time. Since the VM's `machine.yaml` is not tuned, the time is secondary and the state counts are primary.

Ordering features, computed from (G, π) alone:

- Σ_t f_t and max_t f_t, the frontier sizes;
- the **forced-end opportunities** O = Σ_t #{u ∈ F_t : d_U(u) = 1}, the untouched-vertex term in its static form;
- the **low-degree exposure** W = Σ_t #{w ∈ U_t : w has a frontier neighbour and deg_{G[U_t ∪ F_t]}(w) ≤ 2};
- the **sojourn**, i.e. how long vertices stay on the frontier, and the sojourn with exactly one neighbour still to come;
- the work-model prediction for the unpruned work (`work_model.json`).

Analysis:
1. How much do Σ|R|, Σ|P| and ρ vary across the pathwidth-optimal orderings at fixed n?
2. Is the ordering that is best unpruned also best pruned? Compare the ranks under Σ|R| and under Σ|P|.
3. How well do the features predict log ρ and log Σ|P|? Use rank correlation and a least-squares fit of log ρ on the features.

## Experiment 3: refitting the work model with pruning (later)

If experiment 2 shows that the ranking of orderings changes under pruning, refit `work_model.json` on pruned runs, adding the features above, and check whether kpw then selects faster orderings. This is not part of this round.

## Honest limits

- All statements about prediction are empirical, for G_n with n ≤ 56 and orderings from the kpw DAG.
- Uniform samples from the DAG are uniform over the DAG's paths. They are not uniform over all pathwidth-optimal orderings, since the DAG contains only the orderings reachable under its greedy rules.
- VM timings are noisy. The state counts are deterministic.
