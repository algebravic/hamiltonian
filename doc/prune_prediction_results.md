# Predicting the pruning gain: results of experiments 1 and 2

Design: `doc/prune_prediction_design.md`. Code: `hamiltonian/claude/prune_study.py`. Raw data and the generated tables are in `data/prune_study/`: `exact_*.json`, `cands_*.json`, `ord_*.json` and `analysis.md`.

All runs were in the Linux VM (4 threads). The state counts are deterministic; the times are single noisy runs and are not used for any conclusion.

## Notation

- R_t: the reachable states (DP without pruning).
- P_t: the pruned states (DP with pruning).
- L_t: the live states (on some Hamiltonian path).
- ρ = Σ|R_t| / Σ|P_t|: the gain from pruning.
- gap = Σ|P_t| / Σ|L_t|: what any further necessary test could still remove.
- ceiling = ρ · gap: the most that any pruning could gain.

## Validation

The Python reference DP reproduces the C DP's `states_out` step for step at n = 36, 40 and 44, both with and without pruning. Its totals equal A071983(n) for every n in experiment 1, and L_t ⊆ R_t was asserted at every step. Every count in experiment 2 equals A071983(n) as well.

## Experiment 1: exact live sets (work-optimal kpw ordering, n = 30–42 and 44)

The independence predictor and the term shares are explained below the table.

| n | ρ | gap | ceiling | dead % of R | independence prediction % | ρ predicted | share of dead states with s > 2 |
|---|---|---|---|---|---|---|---|
| 30 | 3.31 | 2.16 | 7.15 | 69.8 | 65.4 | 2.89 | 0.54 |
| 31 | 1.90 | 1.61 | 3.06 | 47.3 | 45.3 | 1.83 | 0.41 |
| 32 | 2.01 | 1.69 | 3.40 | 50.2 | 47.3 | 1.90 | 0.31 |
| 33 | 1.74 | 1.53 | 2.65 | 42.5 | 41.2 | 1.70 | 0.35 |
| 34 | 1.79 | 1.34 | 2.39 | 44.0 | 41.9 | 1.72 | 0.33 |
| 35 | 1.58 | 1.22 | 1.92 | 36.7 | 35.2 | 1.54 | 0.43 |
| 36 | 1.81 | 1.22 | 2.20 | 44.7 | 43.4 | 1.77 | 0.54 |
| 37 | 1.99 | 1.49 | 2.96 | 49.6 | 48.0 | 1.92 | 0.50 |
| 38 | 2.29 | 1.57 | 3.58 | 56.3 | 54.7 | 2.21 | 0.65 |
| 39 | 2.45 | 1.63 | 4.00 | 59.2 | 58.4 | 2.41 | 0.68 |
| 40 | 2.64 | 1.34 | 3.55 | 62.1 | 60.5 | 2.53 | 0.61 |
| 41 | 2.49 | 1.54 | 3.82 | 59.8 | 58.9 | 2.43 | 0.63 |
| 42 | 2.75 | 1.92 | 5.29 | 63.6 | 62.9 | 2.70 | 0.72 |
| 44 | 2.28 | 1.33 | 3.04 | 56.1 | 55.6 | 2.25 | 0.75 |

### Findings

**1. The pruned table is exactly the set of reachable states that pass the tests.** At every step of every n above,

|P_t| = #{σ ∈ R_t : σ passes (C) and (D) at step t}.

The difference is 0 at every step, not merely small. This was not guaranteed: in principle a state could pass the tests while being reachable only through states that failed earlier. Two consequences follow:

- ρ = Σ|R_t| / Σ|R_t|(1 − d_t), where d_t is the fraction of R_t that fails the tests. **Predicting the gain is therefore exactly predicting the dead fraction d_t of the reachable states.** No separate model of how earlier pruning propagates is needed.
- A partial explanation can be proved; see the lemma below. It covers the dangling-chain term, which dominates. Exact equality for the other terms is an empirical observation for these n.

**2. Dangling chains are the main cause.** s is the number of dangling chains: chains whose other end has already been eliminated, so that it is a final end of the path. States in which s > 2 on its own already violates the budget account for 31–75% of all dead states. This share grows with n: it is 0.6–0.75 for n ≥ 38. Most of the remaining dead states have s = 2 or s = 1 together with a forced end from the other two terms.

The other two terms are:
- z: untouched frontier vertices with exactly one neighbour still to come;
- w: future vertices whose H-degree is 1.

They rarely exceed the budget on their own (z: 0–7% of dead states; w: about 0). At the largest step, the mean of s is 2.0–2.5, the mean of z is 0.4–1.2, and the mean of w is 0–0.5.

**Lemma (proved).** Along any sequence of valid transitions before the last step, s never decreases.

*Proof.* A dangling chain has one end in the frontier, and its other end is a final end of the path.
- Adding an edge at its frontier end either extends it to an untouched vertex, which leaves it dangling, or joins it to a chain with both ends in the frontier, which gives one dangling chain.
- Joining two dangling chains would close the path, which is invalid before the last step.
- Eliminating its frontier end would also close the path, which is likewise invalid before the last step.
- Eliminating an end of a chain with both ends in the frontier creates a new dangling chain.

So s can only grow. □

Hence a state with s ≥ 3 has only dead descendants, and these all fail the test again. This is why the dead fraction rises along a run: s accumulates as vertices are eliminated. It is also why the table-size ratio grows with the step, reaching 4.9 at the largest step of n = 61.

**3. The independence predictor works.** Take the separate distributions of s, z and w over R_t, treat them as independent, and predict d_t = P(s + z + w > 2 or some future vertex has H-degree 0). This predicts the dead fraction to within 0.5–2.9 percentage points for n ≥ 31, and 4.4 points at n = 30. The gain ρ it implies is within 1–6% of the true value for n ≥ 31, and within 13% at n = 30.

So **the gain is determined, to within a few percent, by the separate distributions of the three endpoint-budget terms over the reachable states.** This holds at n ≤ 44, for one ordering per n. What remains open is predicting those distributions from G and the ordering without running the DP; see "What remains" below.

**4. The tests are close to exact.** The gap between the pruned states and the live states is 1.2–2.2, and mostly 1.2–1.7. No further necessary condition can reduce the states by more than this factor. At these n, that is at most 1.3–1.9× for n ≥ 34. Whether the gap grows with n is unknown: the Python code is too slow beyond n ≈ 44.

## Experiment 2: spread over pathwidth-optimal orderings (n = 50, 52, 54, 56)

For each n there are 14 orderings: the work, v7 and v2 optima, the 3 best under v2, and 8 uniform samples from the kpw DAG. All have optimal pathwidth: 9 for n = 50–54 and 10 for n = 56. Orderings with identical results, which are the duplicated v2 ranks, were counted once, leaving 12 per n. Full tables are in `data/prune_study/analysis.md`.

In the table:
- the spreads are max/min over the orderings, of the states summed over all steps;
- the Spearman column correlates the unpruned and pruned rankings;
- ρ vs R correlates the gain with the unpruned state count;
- ρ vs O correlates the gain with the forced-end opportunities O = Σ_t #{u ∈ F_t : u has exactly one neighbour still to come}.

| n | spread, unpruned | spread, pruned | ρ range | Spearman (unpruned, pruned) | Spearman (ρ, R) | Spearman (ρ, O) | best unpruned / best pruned |
|---|---|---|---|---|---|---|---|
| 50 | 2.99 | 2.21 | 1.94–2.66 | 0.96 | 0.81 | 0.67 | opt-v2 / opt-v2 |
| 52 | 2.15 | 1.84 | 1.84–2.45 | 0.96 | 0.81 | 0.87 | v2-rank1 / v2-rank1 |
| 54 | 1.43 | 1.38 | 1.91–2.18 | 0.93 | 0.58 | 0.52 | sample4 / sample6 (3.5% apart) |
| 56 | 1.50 | 1.36 | 2.11–2.38 | 0.89 | 0.74 | 0.65 | opt-work / opt-work |

Pooled over the four values of n (48 orderings, with each quantity centred within its n), the Spearman correlation of log ρ is 0.72 with O and 0.80 with log Σ|R|.

### Findings

**5. The choice of ordering matters.** Among pathwidth-optimal orderings, the total number of states varies by 1.4–3.0× without pruning and by 1.4–2.2× with it. The gain ρ itself varies by about ±15% around its mean at fixed n.

**6. Pruning compresses the differences between orderings.** Orderings that are worse without pruning gain more from it (Spearman of ρ with the unpruned count: 0.58–0.81). The spread between orderings is therefore smaller with pruning.

**7. The ranking of orderings is essentially unchanged.** The Spearman correlation between the unpruned and pruned rankings is 0.89–0.96. The best ordering is the same with and without pruning for 3 of the 4 values of n, and at n = 54 the two best differ by 3.5%.

Consequence: experiment 3, refitting the work model for pruning, should gain little at these n. A model fitted to unpruned work still picks a near-best ordering for pruned runs. It is not a model of the pruned work, however, so its predicted times are now off by the factor ρ.

**8. The static forced-end feature O predicts the direction of ρ, but not well.** Its Spearman correlation with ρ is 0.52–0.87. Of all the static features tried, it is the best predictor of ρ, but it is too weak to give ρ within a small factor on its own. The low-degree exposure W and the frontier sum Σf are weaker.

## What remains

- **Predicting the distributions of s, z and w from (G, π).** This is what would turn the independence predictor into a structural predictor.
  - The z and w terms depend only on which frontier slots are 0 or −1, and could be modelled per vertex.
  - The s term is a counting problem: the number of path ends eliminated so far, which by the lemma can only grow.
  - A plausible model is a per-vertex probability of being eliminated as an end, estimated from the vertex's degree and the position of its neighbours in π. I have not tried it.
- **How the gap and ρ grow with n.** Measuring this needs a C version of the live-set computation (backward marking). For n ≈ 50–60 it would mean keeping either the tables or the transitions on disk.
- **Wider coverage.** All results are for G_n with n ≤ 56, and the samples are uniform over the kpw DAG, not over all pathwidth-optimal orderings.
