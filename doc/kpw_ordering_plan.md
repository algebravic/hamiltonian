# Replacing MaxSAT + annealing with the kobayashi-pw DAG: measurements and plan

This is the plan for using `kobayashi-pw` in place of the ordering step of `ham_count`: MaxSAT pathwidth via `separation`, then simulated annealing. Per your preference it comes before any code in `hamiltonian` is changed.

- §1 gives the measurements.
- §2 covers what they say about the objective, which turned out to be the main question.
- §3 is the proposed implementation.
- §4 lists what I need you to decide.

**Setup.** Everything was run on a 2-core, 7 GB Linux container:
- `kpw2` compiled from your repo (`gcc -O3 -march=native`);
- your `orders.py`;
- your `ham_ordering.py`, `sa_cost.py`, `sa_cost_params.json` and `ham_dp_c.c`, all unchanged.

**SA orderings.** These are the orderings actually used in the Linux runs. I recovered them from the `vertex` column of `data/linux/profile_*.txt` and checked that each reproduces the logged frontier sizes at every step; all 19 (n = 57–75) do.

**Work.** W = Σ_t states_in(t) · 2^{n_back(t)}. This quantity tracked DP time in the n = 74 analysis, and it does not depend on the machine.

## 1. Measurements

### 1.1 Pathwidth and the DAG: seconds instead of minutes

| n | pw | DAG nodes | orderings in DAG | `kpw2 -c` | Python load | exact V7 optimum | logged MaxSAT + SA |
|---|---|---|---|---|---|---|---|
| 57 | 11 | 13 762 | 1.6·10²⁶ | 0.2 s | 1.5 s | 1.9 s | 67 s |
| 61 | 14 | 263 672 | 2.2·10³³ | 3.9 s | 37 s | 59 s | 930 s |
| 65 | 14 | 112 912 | 2.1·10³³ | 1.9 s | 18 s | 26 s | 329 s |
| 70 | 14 | 110 460 | 4.5·10³⁵ | 1.8 s | 17 s | 27 s | 546 s |
| 74 | 14 | 46 109 | 2.2·10³³ | 1.2 s | 7 s | 10 s | 415 s |
| 75 | 15 | 132 400 | 5.6·10³⁸ | 3.0 s | 23 s | 32 s | 2179 s |

- `kpw2` was run with 2 threads.
- The last column is `ord=` from the logs on the 64-core machine; it includes 1284 s of MaxSAT at n = 75.
- kpw's pathwidth agrees with MaxSAT for every n from 57 to 75.
- `orders.optimize` with `make_model_v7()` reproduces `ham_ordering._dp_cost` to all printed digits for every n, with max frontier exactly pw.

### 1.2 Real DP runs on three orderings per graph

The table compares three orderings:
- **SA**: the logged MaxSAT + SA ordering;
- **V7**: the exact V7 optimum over the DAG;
- **fit**: the exact optimum over the DAG of a *fitted work model* (§2.2).

For each n the fit uses only the runs for the other values of n. All runs use the same DP code (current `ham_dp_c.c`, sort-merge, 2 threads) on the same machine. Every completed run gives the correct H(n).

| n | W (SA) | W (V7) | W (fit) | fit / SA | peak states SA / V7 / fit | DP time SA / V7 / fit |
|---|---|---|---|---|---|---|
| 57 | 4.56·10⁸ | 4.31·10⁸ | **2.52·10⁸** | **0.55** | 5.1M / 4.5M / 4.0M | 83 / 80 / 26 s |
| 58 | 2.21·10⁹ | 1.84·10⁹ | **9.63·10⁸** | **0.44** | 24.4M / 25.2M / 15.0M | 217 / 194 / 103 s |
| 59 | 2.72·10⁹ | > 3.25·10⁹ (†) | **1.76·10⁹** | **0.65** | 87.5M / > 80M / 43.4M | 293 / — / 196 s |
| 60 | 7.47·10⁹ | 8.48·10⁹ | > 3.40·10⁹ (‡) | ? | 98.8M / 88.2M / ≥ 75.5M | — / 941 / — |

(†) Ran out of memory at step 45 of 59; the work up to that point already exceeds the SA ordering's total.

(‡) Ran out of memory at step 43 of 60, twice; the second attempt used 4M-entry worker buffers. The V7 ordering completed on the same container, so the fitted ordering needs more memory somewhere after step 42 than the V7 ordering does. Through step 42 the three orderings have done 4.62·10⁹ (SA), 4.21·10⁹ (V7) and 3.40·10⁹ (fit) of work. SA's remaining 17 steps cost 2.85·10⁹, so the n = 60 comparison is **inconclusive**.

- **SA rows.** The SA rows reproduce the logged W exactly where I reran them (n = 57, 58, 59). The n = 60 entry is the logged value.
- **Timings.** The DP times are noisy, because other jobs shared the two cores during some runs. W and the peaks are exact.

**Reading.** The V7 optimum is not reliably better than SA: it wins at n = 57 and 58 and loses at n = 59 and 60. The fitted-model optimum is better than SA by 1.5–2.3× in work at n = 57, 58 and 59, with lower peaks. That includes n = 59, where V7 lost. n = 60 is inconclusive, because the run could not finish here. n ≥ 61 is untested, because I cannot run those DPs here. The fitted model knows nothing about memory, and the n = 60 run shows that its optimum can need more memory than another ordering even when its work is lower.

## 2. The objective

### 2.1 The current annealing objective is the MLP, and it is unreliable

When `sa_cost_params.json` is present, `sa_refine_order` minimises `sa_cost.sa_cost_fn`, not V7. That is the learned MLP: two hidden layers, autoregressive inputs `log_si` and `prev_log_ratio`, a log-sum-exp aggregate, and the ext penalty when `ram_gb` is set. Three things argue against it:

- **Correlation with actual work.** Over n = 62–74 (13 graphs, all width 14), the correlation of log W with the MLP cost of the ordering actually run is 0.34; with log V7 it is 0.75.
- **The n = 75 ordering.** The MLP gives that ordering a cost of 10.5, the lowest of all 19 logged orderings. It was by far the most expensive run (W = 5.1·10¹¹, 3.9 times n = 74). V7 rates it 12 times worse than the DAG's V7 optimum. The annealing found an ordering that the network mispredicts.
- **The ext penalty never fires.** With `ram_gb = 460`, every MLP cost above is identical to `ram_gb = 0`. The network's predicted state counts (about e^17 ≈ 2·10⁷) are far below the real peaks (8·10⁸), so `_ext_log_overhead` is never triggered.

There are caveats:
- the n = 62–74 runs used an earlier version of the MLP (the JSON is dated 18 April, the runs 11–13 April);
- the sample is small;
- every ordering in it was chosen by SA.

The MLP is also not step-local, so it cannot be optimised exactly over the DAG.

### 2.2 A fitted, step-local work model

The DP's work is W = Σ_t s_t·2^{nb_t}, with s_{t+1} = s_t·exp(g_t), where g_t = log(states_out/states_in). If g_t is predicted from features of the step alone, that is, from the prefix and the vertex added, then W has exactly the (a, e, k) form of `orders.py`:

    a = 0,   e = 2^{nb},   k = exp(ĝ) / 2^{nb}     (cost += p·e;  p ← max(p·e·k, 1))

The Pareto-label argument applies, because the remaining cost is nondecreasing in p when e ≥ 0. So the predicted work can be minimised **exactly** over the DAG.

**The fit used for §1.2:**
- ordinary least squares on the 925 logged steps with states_in ≥ 1000, from all 19 Linux runs;
- 11 features, all of them functions of the prefix and the vertex added (`orders.StepFeatures`):
  - n_back, n_elim, fw, fw before the step, e_bag;
  - Σ log Bell(component sizes);
  - nb·fw, n_elim·fw_before, nb², step/n, and a constant;
- R² = 0.887, and 0.880 when each n is held out in turn; per-step RMSE 0.42 in log units.

**Absolute predictions are poorly calibrated:**
- along a whole ordering, the per-step errors compound;
- the predicted W of the SA ordering at n = 57 is 2.4·10¹⁰, against 4.6·10⁸ actual;
- the model's own optimum at n = 57 is predicted at 4.5·10⁶, against 2.5·10⁸ actual.

So the model ranks orderings well, but its numbers should not be used for memory or spill decisions. The optimiser probably also exploits the regions where the model extrapolates; §1.2 shows the resulting orderings are still good.

**Cost of exactness.** V7 never produced more than one Pareto label per node. The fitted model produces up to 366, 716, 905 and 665 labels (n = 57–60), and the Python optimiser takes about 4–6 minutes at n = 59–60. A C optimiser in kobayashi-pw would make this a few seconds; that is now worth doing for n = 75.

**It improves with use.** Every new run logs its per-step state counts, and those runs use orderings chosen by the model itself. Refitting on them is the natural guard against the model being exploited.

## 3. Proposed changes

### 3.1 Dependency

In `hamiltonian/pyproject.toml`:

```toml
dependencies = [ ..., "kobayashi-pw" ]
[tool.uv.sources]
kobayashi-pw = { path = "../kobayashi-pw", editable = true }
```

An editable path install keeps `kobayashi_pw/../C/kpw2` resolvable, and that is how `orders_cli._find_kpw` locates the binary. A git source would install the package without the compiled programs, and the code would then fall back to building the DAG in Python: 155 s instead of 4 s at n = 61. The binary can also be given explicitly with `--kpw-bin` or the config key `kpw_bin`.

`separation` stays a dependency, because `--ordering maxsat`, `profile_analysis.py` and `generate_diverse_profiles.py` use it.

### 3.2 New modules (relative imports throughout)

**`hamiltonian/claude/ham_kpw.py`**
- `adj_to_graph(adj, n)` returns a `kobayashi_pw.Graph` with labels 1..n.
- `load_dag(adj, n, *, cache_dir, kpw_bin, threads, greedy2=False)`:
  - runs `kpw`/`kpw2 -c -o` through `orders_cli.build_dag`;
  - caches the result as `data/dags/G{n}_{edgehash}.dag`, keyed by a hash of the edges;
  - a cached DAG loads without running kpw.
- `kpw_order(adj, n, *, cost="work", work_model=None, expand_base=1.55, density_alpha=0.25)` returns `(pw, order, info)`:
  - the order is the exact optimum over the DAG, in the frontier-DP direction;
  - before returning, it checks that the order's max frontier equals pw; for V7 it also checks agreement with `_dp_cost`.
- `kpw_candidates(...)` returns candidates for multi-start or reranking:
  - the optima under work, V7 and V2;
  - the k best under V2 (`orders.k_best`);
  - optionally uniform samples (`dag.sample`).

**`hamiltonian/claude/work_model.py`**
- The step features and the (a, e, k) model of §2.2, with coefficients read from `work_model.json`.
- `fit_work_model(log_paths) -> json`, which reads `profile_*.txt` files and recovers each ordering from the `vertex` column. It records which runs were used and reports the held-out R².
- Also exposed as `uv run fit-work-model data/linux/profile_*.txt -o hamiltonian/claude/work_model.json`.

### 3.3 Changes to `ham_count.py`

- **`--ordering {maxsat,kpw,bfs}`** (config key `ordering`). `--no-pw` remains an alias for `bfs`.
- **kpw options:**
  - `--kpw-cost {work,v7,v2}`, default `work`, falling back to `v7` with a warning if no `work_model.json` exists;
  - `--work-model PATH`;
  - `--kpw-bin PATH`;
  - `--kpw-threads N`, default 0 (all cores);
  - `--dag-cache DIR`, default `data/dags`;
  - `--kpw-greedy2`, off by default (smaller DAG, worse orderings).
- **Refinement after kpw.** None by default: the DAG optimum is used as is. `--kpw-polish` runs the existing `sa_refine_order` from it.
- **`--kpw-rerank {none,partial-dp}`.** Ranks `kpw_candidates` with the existing `validate_multistart_orders` (a partial DP over the first K steps).
- **`--multi-start` with `--ordering kpw`** uses `kpw_candidates` instead of enumerating MaxSAT solutions.
- **Logging.** The header line reports `ordering=kpw-dag(work)`, the DAG size, the number of orderings, and the build, load and optimise times.

`ham_ordering.py`, `sa_cost.py` and the C code are unchanged.

### 3.4 Tests (`tests/test_kpw.py`)

1. For n = 15–45, kpw's pw equals the max frontier of its ordering, and equals MaxSAT's pw when `separation` is installed.
2. For n = 15–40, the DP count with each kpw ordering (work, V7, V2) equals `data/b071983.txt`.
3. Each optimum is ≤ the same cost on 100 uniform DAG samples, and `k_best` under V2 is sorted and starts at the V2 optimum.
4. `fit_work_model` on two logs is deterministic, and its exact optimum equals brute-force enumeration of DAG paths on a small graph.
5. The DAG cache returns the same DAG on a second call, and a different edge set gives a different key.

### 3.4a Memory: of the counting DP, and of the optimiser (added after review)

**Which memory ran out.** The out-of-memory kills at n = 59 and 60 were in the Hamiltonian-path DP (`ham_dp_c`), not in the ordering optimisation.

At n = 60 the fitted ordering enters step 43 with 64M states and adds a vertex with at least 3 back-edges. That step produces up to about 64M·2³ ≈ 5·10⁸ raw contributions before deduplication, about 8 GB at 16 bytes each, so it spills to the backing store, and the run was killed during it.

The V7 ordering survived a step of almost the same raw size: step 42, with 62.6M states and 3 back-edges. That step's output was only 27M states, because of eliminations. So the memory a step needs depends on both states_in·2^{n_back} (the raw contributions) and the size of its output. The fitted ordering's output at step 43 is unknown, because it was killed. Neither quantity is penalised by the work objective.

**Memory of the optimiser.** The optimisation over the DAG is a dynamic program over the DAG's nodes, and it is linear in the DAG, as you expected from BDDs. But each node stores a *set* of labels, not one value:

- **Additive costs (V2) or proxies whose clamp never binds (V7 here):** one label per node, so memory ≈ DAG size. Measured at n = 58: 95 MB after loading the DAG (45 056 nodes, 320 888 arcs, held in memory), 110 MB after optimising.
- **The work model:** a label is (cost so far, predicted state count p). Two partial orderings reaching the same node, one cheaper so far and one with fewer predicted states, are both kept, because either can win later. That gives up to 366–905 labels at a node (n = 57–60) and 826 MB at n = 58 in Python, with all fronts kept for traceback. In C, a label is 24 bytes (cost, p, back-pointer), about 30× smaller.

**Variants, in order of preference:**

1. **Exact optimum under a cap on per-step size.** Drop every label whose predicted states_in·2^{n_back}, or predicted states_out, at the next step exceeds a cap C. Dropping by a criterion that is monotone in p keeps the Pareto argument valid, so the result is the exact minimum of predicted work subject to the cap. It also prunes exactly the high-p labels that make the fronts large, so the optimiser uses less memory as well.

   Because the model's absolute predictions are unreliable, C should be relative:
   - first compute B, the smallest achievable maximum of predicted states·2^{n_back} along a DAG path. This is a bottleneck DP over the DAG, using labels (max so far, p).
   - then set C = B·r, with r = 1 giving the minimum-memory ordering, and larger r trading memory for work.

   Option: `--kpw-mem-slack r`.

2. **Spill penalty instead of a cap.** Multiply e by about 2.5 at steps whose predicted states·2^{n_back} exceed a threshold. This is your ext factor, and it is still monotone in p, so the optimum is still exact. It has the same calibration problem, so I would add it only after refitting.

3. **Bounded optimiser memory (`--kpw-max-labels L`).** Keep the L cheapest labels per node. `orders.optimize` already supports this. The result is suboptimal, with optimiser memory at most DAG size × L, and the result reports whether truncation happened.

4. **Lower-memory DAG.** `-g` (second greedy rule) shrinks the DAG 2–6×, but your earlier measurements show its orderings are noticeably worse, so I would not use it for this.

Variants 1 and 3 go into `ham_kpw.py` and the C optimiser. Variant 1, with the slack r = ∞ (no cap) by default and r set per machine in the run config, is the one I recommend, since that is where the 768 GB / 7 GB difference belongs.

### 3.5 For the H(75) recheck

The recheck needs a different optimal ordering at n = 75. The DAG offers about 5.6·10³⁸ of them in 3 s, and the fitted model's optimum at n = 75 is a natural candidate, possibly with less work and a lower peak than the 768 GB run. Its work prediction is not reliable enough to size the machine with, though.

### 3.6 Documents

Once the code is in and you have run it at scale, the engineering paper and talk 2 should describe how orderings are produced. That means the MaxSAT section, the tuning paragraph, and the ordering times quoted in Results. I would not change them before then.

## 4. Decisions for you

1. **Default objective for `--ordering kpw`.** I recommend `work`, the fitted model, given §1.2; `v7` is the fallback. A memory term would be a natural extension, such as a penalty on predicted states above a threshold, as `sa_cost` attempted. It needs the absolute calibration that the model lacks, so I would add it only after refitting on runs that used its own orderings.
2. **Default `--ordering`.** Keep `maxsat` until you have compared at n ≥ 61 on a large machine, or switch to `kpw` now? §1.1 suggests the switch is safe for pw, because kpw and MaxSAT agree on every n tested.
3. **Dependency.** The editable path source `../kobayashi-pw` (recommended), or the git source plus `kpw_bin`?
4. **A C optimiser for (a, e, k) models in kobayashi-pw.** It would reduce the 4–6 minutes of Python per n (more at n = 75) to seconds. It would be a change to kobayashi-pw, with its own plan.
5. **The MLP annealing path.** Leave it as is, behind `--ordering maxsat`, or retire it once kpw is the default?

## 5. Implemented: first version (no memory cap)

**New files**
- `hamiltonian/claude/ham_kpw.py`:
  - builds the DAG with `kpw`/`kpw2 -c` and caches it in `data/dags/G{n}_{edgehash}.dag`;
  - minimises `work`, `v7` or `v2` exactly over it with `kobayashi_pw.orders.optimize`;
  - checks every returned ordering: it is a permutation, its max frontier is pw, and for V7 its value agrees with `_dp_cost`;
  - `kpw_candidates` gives the optima under each cost, the V2 k best, and optional samples.
- `hamiltonian/claude/work_model.py`:
  - the fitted model of §2.2;
  - log parsing: the ordering comes from the `vertex` column, n from the number of steps, and a log is used only if its frontier column agrees with G_n; duplicate orderings are used once;
  - the fit, with the leave-one-n-out R²;
  - the `fit-work-model` command.
- `hamiltonian/claude/work_model.json`, fitted on the 19 distinct Linux runs (n = 57–75): 925 steps, R² 0.887, held-out R² 0.880.
- `tests/test_kpw.py`, 50 tests:
  - pw-optimality for every cost;
  - pw equal to MaxSAT's, for n = 15–40;
  - DP counts equal to `data/b071983.txt`;
  - each optimum ≤ 50 uniform samples;
  - V7 equal to `_dp_cost`;
  - the work optimum equal to brute-force enumeration of all DAG paths, on G₁₀–G₁₂ and small random graphs;
  - candidates, the cache, and the fit (determinism, duplicate and bad-log rejection).

**Changed files**
- `ham_count.py`:
  - options `--ordering {maxsat,kpw,bfs}` (default `maxsat`), `--kpw-cost`, `--work-model`, `--kpw-bin`, `--kpw-threads`, `--dag-cache`, `--kpw-greedy2`, `--kpw-max-labels`, `--kpw-polish` and `--kpw-candidates K`;
  - `--kpw-candidates K` reranks candidates by partial DP over `--multi-start-validate` steps (default n/2);
  - `--no-pw` is now an alias for `--ordering bfs`;
  - the header reports `ordering=kpw-dag(<cost>)`, and a `kpw:` line gives the DAG size, the number of orderings, the build, load and optimise times, and the number of labels;
  - the result line is unchanged, so existing parsers still work;
  - the MaxSAT and SA paths are unchanged.
- `ham_dp_c.py`: fixed a pre-existing bug. `partial_dp_time_c` passed 12 arguments to `count_ham_paths_c`, which has taken 14 since cycle counting was added. So `--multi-start-validate` raised `TypeError`; it now passes `NULL` for the cycle outputs.
- `pyproject.toml`:
  - adds `kobayashi-pw` (editable path source `../kobayashi-pw`) and `numpy`, which `sa_cost.py` already imported;
  - adds the script `fit-work-model`;
  - **run `uv sync` to update `uv.lock`.**

**End-to-end runs** (container, 2 cores; all counts equal `data/b071983.txt`):

| run | ordering time | DP |
|---|---|---|
| n = 30–32, `--ordering maxsat --refine-iters 2000` | 0.8 s each | correct (unchanged path) |
| n = 30–32, `--ordering kpw` | 0.1 s each | correct |
| n = 45, `--config` with `ordering: kpw`, `kpw_cost: v7` | 1.8 s | correct |
| n = 45, `--kpw-polish` | 11 s | correct |
| n = 50, `--kpw-candidates 5` (7 candidates, partial DP over 25 steps; chose the work optimum) | 19 s | correct |
| n = 57, `--ordering kpw --sort-merge`, first and second (cached) run | 17.6 s / 17.4 s | 29 s |

The n = 57 ordering time is dominated by the Python optimiser (16 s, 451 labels); the C optimiser will remove most of it.

**Known cosmetic issue.** With `--kpw-candidates`, the existing `validate_multistart_orders` prints `sa_cost=0.000e+00` for each candidate, because kpw candidates carry no SA cost. The chosen candidate is reported by name on the next line.

## 6. The C optimiser (kopt) and the model's dependence on the backend

### 6.1 kopt

`kobayashi-pw/src/C/kopt.c` runs the same Pareto-label DP as `orders.optimize`, on the DAG file. It reproduces the Python expansion order, so the blocks and step features are the same, and it breaks ties the same way.

- **Agreement.** On the square-sum graphs (n = 40–57, all three costs) it returns the same orderings as Python, with costs equal to about 1e-15. It also matches brute-force enumeration of DAG paths on random graphs.
- **The work model is shared.** `orders.py` now defines the work-model features once (`WORK_FEATURES`, `work_features`, `make_model_loglin`), and `work_model.py` uses them. So the fit, the Python optimiser and `kopt -m loglin` agree.
- **`ham_kpw` changes:**
  - it uses kopt whenever the binary exists;
  - it reads only the DAG file's header;
  - it loads the DAG into Python only for `--kpw-candidates` (k best and samples) or with `--kpw-python`.
- **New `ham_count` options:** `--kpw-python` and `--kopt-bin`.

| n | before (Python) | now (kpw + kopt) |
|---|---|---|
| 57 | 17.6 s | 0.4 s |
| 61 | several minutes (not measured; 4–6 min at n = 59–60) | 15.7 s (3.2 s DAG, 12.5 s kopt; 616 MB) |
| 75 | not run | 11.3 s (2.9 s DAG, 8.3 s kopt; 411 MB) |

For comparison, the logged MaxSAT + SA ordering times were 930 s at n = 61 and 2179 s at n = 75.

Build it with `make -C src/C kopt` in kobayashi-pw. The tests are `tests/test_kopt.py` in kobayashi-pw and `test_c_and_python_optimisers_agree` in `tests/test_kpw.py`.

### 6.2 Does the work model depend on the backend (extendible hashing vs sort/merge)?

**What the model predicts does not.** It predicts g_t = log(states_out/states_in). The states at each step are the distinct keys after the step, which depend only on the graph and the ordering. Both backends compute exactly the same table. So logs from either backend, on any machine, are valid training data, and runs with the same ordering add nothing new (`fit-work-model` drops them). W = Σ s_t·2^{nb_t}, the number of (state, subset) contributions generated, is likewise backend-independent.

**How W turns into time does depend on the backend:**
- **Extendible hashing:** every contribution is a hash insertion, so the time is roughly proportional to W. Memory is the current and next tables; it is proportional to the states, with no spill.
- **Sort/merge:** every contribution is written to a buffer and radix-sorted, so the time is again roughly proportional to W. On top of that, a step spills to SSD when its raw output does not fit (§3.4a), and that costs 2–3× more. In the first n = 74 run, about 40% of the old code's time (1417 of 3449 s) was not proportional to W at all: the post-introduce re-sort, which the current code no longer has.

So the same fitted model serves both backends as a predictor of work. A backend-specific term would belong in the *cost* rather than the model: a spill penalty or cap for sort/merge (variants 1–2 of §3.4a), or a cap on states for extendible hashing, whose memory is proportional to the table. Neither is in this version.

## 7. The DSL engine (`--kpw-engine dsl`)

kobayashi-pw's DSL can now express the secondary-cost optimisation (`doc/dsl_secondary_design.md` there). `ham_count --ordering kpw --kpw-engine dsl` uses it.

- **How it differs from the kopt path.** In one process, generated C does the complete search, the pruning and the Pareto optimisation, with no DAG file. The V2, V7 and work costs come from `kobayashi_pw.dsl.hampath`, with the work model's coefficients read from `work_model.json` and passed as runtime parameters, so refitting does not recompile.
- **Same results.** It gives the same orderings as the kopt path (`tests/test_kpw.py::test_dsl_engine_equals_kopt`, n = 30, 40, 50, all three costs). On G₄₀–G₇₅ the orderings are identical to kopt's.
- **Speed.** It takes 0.66–1.02× the time of `kpw -c` + kopt (kobayashi-pw `doc/dsl_codegen_perf.md`). At n = 57 the whole ordering step took 0.39 s, against 0.43 s for the kopt path.
- **Default.** kopt stays the default. `--kpw-candidates` still loads the kpw DAG, because it needs `k_best` and sampling, which the DSL does not provide yet.
