# Making the ordering cost model trustworthy

Status: design, not implemented.

## What the model is for

`--ordering kpw` finds the pathwidth-optimal orderings and then picks among them the one minimising a secondary cost. The default cost is the work model in `work_model.json` (`hamiltonian/claude/work_model.py`). It predicts the per-step log-growth g_t = log(states_out / states_in) as a linear function of 11 step features:

- const, n_back, n_elim, fw, fw_before, e_bag, log_bell_components;
- n_back·fw, n_elim·fw_before, n_back², step/(n−1).

From g_t it forms W = Σ_t s_t·2^{nb_t}, which is minimised exactly over the DAG. **The only use of the model is to rank orderings.** Its absolute predictions are documented as poorly calibrated.

## What is wrong with the current fit

1. **Paucity.** It was fitted from 9 logs, all of G_60: 407 steps in total, but only 9 orderings of a single graph.
2. **No out-of-sample check.** `r2_leave_one_n_out` = 0.894 equals the in-sample R², because there is only one n to leave out. The generalisation to other n has never been measured.
3. **Bias.**
   - The logs are unpruned runs, but the DP now prunes, and pruning changes the per-step growth by factors of 2–5.
   - The orderings came from earlier SA and MaxSAT experiments. They are not representative of the pathwidth-optimal orderings the model now ranks.
4. **Wrong criterion.** The fit minimises the squared error of g_t per step. What matters is the regret of the ordering it selects: how much worse that ordering is than the best available one.

**The cost.** In experiment 2 (`doc/prune_prediction_results.md`), the work-optimal ordering was best in total pruned states only at n = 56. At n = 50, 52 and 54, other pathwidth-optimal orderings beat it by 4%, 9% and 18%.

## Plan

### Phase 1: data and honest validation (do first)

**Data.** For each n ∈ {44, 46, 48, 50, 52, 54, 55, 56}, take the orderings of `ham_kpw.kpw_candidates`:
- the optima under work, v7 and v2;
- the 3 best orderings under v2;
- 16 uniform samples from the kpw DAG (fixed seed).

Run the C DP on each, with `--prune drop` and with `--prune none`, collecting per-step `states_in`, `states_out` and the contributions written. This reuses `prune_study.py orderings`, extended to record per-step tables and not only totals. The 14-ordering data for n = 50–56 already exists in `data/prune_study/`.

Optionally, you could add n = 58–60 on the Macs; in the VM the unpruned runs at n = 58 take more than 3 minutes each.

**Targets.** For each ordering π and mode m ∈ {drop, none}:
- W_m(π) = Σ_t s_t · 2^{nb_t}, the quantity kpw minimises;
- G_m(π) = Σ_t (contributions written at step t), the work actually done.

The primary target is W_drop, because the DP prunes by default.

**Validation.** Leave one n out: fit on all other n, and evaluate on the held-out n with:
- **rank correlation:** Spearman between predicted W(π) and actual W_drop(π) over that n's orderings;
- **regret:** W_drop(model's best) / min_π W_drop(π), over the sampled orderings;
- **DAG regret:** W_drop of the model's exact optimum over the whole DAG, divided by the best sampled W_drop. This is the regret that matters in production. It can be below 1, if the model finds an ordering better than any sample.
- **per-step RMSE of g_t**, reported but not used to choose between models.

**Baselines,** each evaluated the same way:
- the current `work_model.json`;
- v7 and v2;
- Σ_t f_t (the frontier sum);
- the refit 11-feature model, fitted on pruned data across many n.

**Code changes.**
- `work_model.py` fits on several n and on pruned logs. It writes `r2_leave_one_n_out` and `rmse_leave_one_n_out` only when at least two distinct n are present (null otherwise), and records the list of n and the modes in the metadata.
- A new `evaluate_cost_models(data_dir)` in `prune_study.py` produces the validation table.

**Decision rule.** Adopt the refit model as the default if its median DAG regret over the held-out n is lower than the current model's, and it is worse by more than 5% at no held-out n. Otherwise keep the current default and record why.

### Phase 1b: the per-step cost multiplier (added 2026-11-27)

The work proxy W = Σ_t s_t·2^{nb_t} charges each input state 2^{n_back}, the number of back-edge subsets the sweep used to enumerate. Two facts change this.

1. **Only subsets of size ≤ 2 can be valid (proved).** The sweep now enumerates only those: 1 + nb + C(nb, 2) masks (`doc/subset_enumeration_design.md`). So 2^{nb} no longer describes the generation cost.
2. **The outputs per input barely depend on n_back.** At n = 61, steps with ≥ 20 M input states wrote, on average:

   | n_back | pruned | unpruned |
   |---|---|---|
   | 1 | 1.67 | 1.85 |
   | 2 | 1.69 | 2.24 |
   | 3 | 2.03 | 3.10 |
   | 4 | 0.42 | 0.66 |

   (the nb = 4 row is a single step).

   Before the change, the generation time per input state grew by a factor of about 1.6–2.6 per back-edge, because all 2^{nb} subsets were enumerated.

The cost is therefore better written as two terms:

  e_t = α·m(nb_t) + β·o_t,

where m(nb) = 1 + nb + C(nb, 2) is the number of masks per input state and o_t is the expected number of outputs per input state. Generation pays for the masks; flushing, merging and memory pay for the outputs.

The exact DAG optimiser accepts any per-step multiplier e_t ≥ 0 of the carried state count (the e of the (a, e, k) form), so neither term needs to be of the form c^{nb}.

**Fitting α and β:**
- α comes from the per-step generation times divided by the input states, in the instrumented logs.
- β comes from the flush and merge times divided by the contributions written.
- o_t is modelled like g_t, or with a small table indexed by n_back.

A free table e(nb), fitted directly to the per-step times, is the model-free alternative.

**Validation** is as in Phase 1, with the measured time (generate + flush + merge) as an additional target besides the state counts. That requires runs with real memory: on the VM every run spills, so these time fits are for the Macs.

### Phase 2: a model built from the DP's mechanics (if Phase 1 leaves real regret)

Replace the free 11-term regression by a growth model with few parameters, each of which has a meaning:

g_t = log E_t − log D_t − log S_t, where:

- **E_t, expansion.** The mean number of valid back-edge subsets per input state, out of 2^{nb_t}.
  - A subset is invalid if it touches an interior frontier vertex or closes a cycle. So E_t depends on the fraction of frontier slots that are interior or are chain ends.
  - It is modelled from a per-slot status distribution that evolves with the number of the slot's edges already decided. This is a mean-field model with one or two parameters.
- **D_t, deduplication.** The collapse from merging equal keys after the eliminations.
  - Its natural bound is the ratio of the possible frontier configurations before and after the step (the log-Bell difference already among the features).
  - A coefficient with a prior centred at 1 multiplies that bound.
- **S_t, pruning survival.** The fraction surviving the endpoint budget. It comes from the independence predictor of `doc/prune_prediction_results.md`, using a model of the distributions of s (dangling chains), z and w.
  - By the dangling-chain lemma, s can only grow along a run, which makes it a counting process along the ordering.

The parameters are fitted with **shrinkage towards their mechanistic values**: Bayesian ridge, with the prior means set by the mechanics. This way sparse or biased data can move them only as far as the evidence supports. It is validated exactly as in Phase 1.

This phase is research. Its success is not assured, and it starts only if Phase 1 shows a median DAG regret above about 5%.

### Phase 3: ranking by measurement (alternative or complement)

Rank a short list of candidate orderings by running the actual pruned DP, not by a model.
- `--kpw-candidates K` already ranks candidates by a partial DP over the first k steps.
- **Phase 1 data first answers whether that ranking is reliable:** does the order after k steps predict the order of the full totals, and for which k?
- If it is reliable, the candidate list would be: the model optimum, the v2 k-best, and a few DAG samples.

There is no model to bias here, but the cost is K partial runs.

## Deliverables

1. A validation table (`data/work_model_study/validation.md`): held-out rank correlation, regret and DAG regret for each cost and each n.
2. A refit `work_model.json`, adopted only if it passes the decision rule, with honest metadata.
3. A short results note, `doc/work_model_results.md`, recommending one of: keep the current model, adopt the refit, proceed to Phase 2, or switch to measurement-based ranking (Phase 3).

## Honest limits

- The validation covers G_n for n ≤ 56–60 and the orderings in the kpw DAG. Extrapolating to n = 70–75 is not validated.
- The "best" ordering is the best of about 20 samples, not the true optimum over the DAG. Regret is therefore measured against a sample, and true regret could be larger.
- The VM has 3 GB of RAM, so its timings involve spilling. The targets are state counts, which do not depend on the machine.

## Driver (implemented 2026-11-27): `hamiltonian/claude/ordering_study.py`

1. **Candidates, once, on one machine.**
   ```
   python -m hamiltonian.claude.ordering_study candidates 58 66 \
       --models work_model.json data/linux/linux_work_model.json sri_work_model.json work_m2pro.json
   ```
   - This writes `data/ordering_study/cands_<n>.json` for each n.
   - Each file holds the exact DAG optimum under each model file, the v7 and v2 optima, the 3 best under v2 and 16 uniform DAG samples.
   - It also stores every model's predicted work for every candidate, and the md5 of each model file.
   - Generate these files on a machine that has the kpw binaries; the Linux box is fastest. In the VM, without them, one n took more than a minute. Copy or commit the files so that all machines run identical orderings.
2. **Runs, on every machine, resumable.**
   ```
   python -m hamiltonian.claude.ordering_study run 58 66 --machine-yaml <yaml> [--host NAME]
   ```
   Each run is the pruned DP with `-v --instrument`. It writes `data/ordering_study/<host>/run_<n>_<tag>.json` containing:
   - the per-step table;
   - the per-step generate, flush, merge and introduce times;
   - the contributions written, the buffer size and the spills;
   - the totals, and whether the count matches `data/b071983.txt`.

   The first run on a host writes `host.json`, the machine's fingerprint: CPU, cores (P/E on macOS), RAM, cache sizes and sharing, and the `machine.yaml` contents. Each run's header lines record the compiler, the build key and the `SM_*` settings. Existing results are skipped, so the driver can be stopped and restarted at any time.
3. **Analysis.** `python -m hamiltonian.claude.ordering_study analyze` writes `data/ordering_study/analysis.md` with:
   - for each n and each model: the regret of its DAG optimum and of its best-predicted candidate, and its Spearman rank correlation with the actual Σ states (these are machine-independent);
   - for each host, per-step cost coefficients in ns per unit, fitted on steps with ≥ 1 M input states:
     - a: generation per mask (s_in × (1 + nb + C(nb, 2)));
     - b: generation per output;
     - c: flush per output;
     - d: merge per run entry;
     - plus the overall ns per state.

**Correlating with hardware.** The coefficients a–d are the machine-dependent part of the cost; the state counts are not machine-dependent. Comparing a–d across hosts, against the fingerprints (P-core count and frequency, memory bandwidth, L2/LLC size, NUMA), tests which resource each phase is limited by:
- generation should scale with core count and frequency;
- flushes and merges with memory bandwidth;
- the merge also with LLC size, through the number of streams.

Memory bandwidth is not yet in the fingerprint. `src/C/cacheprobe.c` or a STREAM-like probe could add it.

**First check (VM, n = 44, 10 candidates, Σ states with pruning):**

| model | regret of its optimum | Spearman |
|---|---|---|
| `work_model.json` | 1.09 | 0.70 |
| `linux_work_model.json` | 1.50 | 0.49 |

At n = 64 it was the other way round: the Linux model's ordering had 1.6× fewer states. One small n settles nothing, which is the point of running n = 58–66.
