
markdown

# Plan: Pushing the Hamiltonian-Path Count for G_n Beyond n = 56

## Background and Bottleneck Diagnosis

### The graph G_n
G_n has vertices {1, …, n}. There is an edge (i, j) iff i + j is a perfect square.  
Key structural facts:
- Every vertex has degree ≤ 7 for n ≤ 100 (sparse graph).
- The graph is connected for n ≥ 15 (it has at least one Hamiltonian path for 15 ≤ n ≤ 17, etc.).

### Why graphillion runs out of memory

graphillion represents sets of subgraphs as a Zero-suppressed Decision Diagram (ZDD).  
The ZDD processes the edges of the graph **one at a time**, in the order defined by the
universe list.  At each edge step it remembers every possible "profile" of the partial
solution across the **frontier** — the set of vertices that have at least one already-
decided edge and at least one not-yet-decided edge.

The total ZDD node count is roughly

    ∑_{edge steps}  |live profiles at that step|

For a profile width (frontier size) W, the theoretical maximum number of live states is
of order 3^W (three choices per frontier vertex: interior, endpoint-of-segment-A,
endpoint-of-segment-B, …), but ZDDs also memoize across all prefix choices and so the
constant can be much larger.

Empirically, going from a vertex ordering with max frontier width 39 (natural 1, 2, …, n
order) to width 17 (best-BFS order, see §1 below) reduces peak ZDD memory by several
orders of magnitude; but for n > 56, even width 17 eventually exhausts available RAM
because the ZDD size explodes **exponentially** in the frontier width.

---

## Improvement 1 — Better Vertex Ordering (zero-cost, large impact)

### What the ordering controls

graphillion's universe list is a sequence of edges.  The user provides a
*vertex* ordering and derives the edge ordering by sorting edges as

    key(u, v) = ( max(pos[u], pos[v]),  min(pos[u], pos[v]) )

This groups edges so that the vertex that "finishes last" is introduced last.  The
**frontier width at edge step k** equals the number of vertices that have at least one
already-decided edge and at least one future edge.  The maximum frontier width over all
steps equals the **pathwidth** of G_n under that vertex ordering.

### Best-BFS ordering

The current implementation computes an ordering via a pathwidth heuristic.  A much
simpler and often equally good approach is:

1. For every vertex s ∈ {1, …, n}, run a BFS of G_n starting from s.
2. Extend the BFS traversal order to a full ordering (append remaining vertices by
   ascending label if the graph has components, though G_n is connected for n ≥ 15).
3. Compute the **max frontier width** and **profile sum** (sum of widths at all steps)
   for each candidate ordering.
4. Keep the ordering that minimises (max_fw, profile_sum) lexicographically.

This is O(n²) and typically matches or beats a pathwidth heuristic.

Measured results (frontier width = pathwidth estimate):

| n  | Natural order max_fw | Best-BFS max_fw | Reduction |
|----|---------------------|-----------------|-----------|
| 56 | 39                  | 17              | ~2.3×     |
| 60 | (n/a)               | 19              | —         |
| 70 | (n/a)               | 23              | —         |

Because ZDD size grows **exponentially** in max_fw, halving the frontier width can yield
an exponential reduction in memory usage.

### Profile-sum minimisation (local search)

Even with the same max_fw, a smaller **profile sum** (the sum of frontier widths across
all edge steps) directly reduces the total number of ZDD nodes.  After fixing the best
max_fw via BFS, apply a short simulated-annealing local search:

- Moves: random swap of two positions; random reversal of a contiguous segment.
- Objective: minimise (max_fw, profile_sum) lexicographically.
- Temperature schedule: linear from T₀ = 3 to T_f = 0.01 over 100 000 iterations.

In practice this reduces the profile sum by 5–20 % beyond the best BFS order.

---

## Improvement 2 — Replace graphillion with a Direct Profile DP

### Key insight

A ZDD implicitly enumerates all valid partial solutions at every edge step; it keeps
them as a shared DAG.  The peak memory equals the total number of *distinct ZDD nodes*
accumulated across the entire computation, which can be much larger than the number of
*currently live states* at any single step.

A **direct hash-map profile DP** stores, at each edge step, only the set of currently
live states together with their counts.  States that die (unfeasible partial solutions)
are immediately discarded and their memory is freed.  This means peak memory scales with
the number of *simultaneously reachable live states* rather than with the total ZDD DAG.

For this graph, the live state count at the peak is orders of magnitude smaller than the
full ZDD, because:
- The graph is sparse (max degree ≤ 7): most states die quickly.
- The `num_sealed` counter (see §2.2) aggressively prunes multi-component states.

### State representation

The state of the DP at each edge step is:

    ( frontier_profile_tuple, num_sealed )

where `frontier_profile_tuple` is a canonically-labelled tuple of length equal to the
current frontier size, with entries:

- `0`  — vertex is in the frontier but has no selected path-edges yet (unvisited).
- `-1` — vertex is an interior node (two selected path-edges; degree saturated).
- `k > 0` — vertex is an open *endpoint* of path-segment k (one selected path-edge).

Two states with the same **canonical** labelling (segment IDs assigned in
first-occurrence order) are equivalent and their counts are added.

`num_sealed` counts the number of path segments whose *both* endpoints have left the
frontier (both "external").  For a Hamiltonian path there must be exactly `num_sealed = 1`
at the end.  States where `num_sealed > 1` are discarded immediately (they represent
disconnected multi-path subgraphs, which can never become a single Hamiltonian path).

### The bug that must be avoided (and is fixed here)

There is a subtle case that is easy to get wrong: when two segments are **merged** via a
new edge, and *both* of their "other" endpoints are already external (forgotten), the
merged segment immediately seals without any vertex being forgotten at that moment.  The
`num_sealed` counter must be incremented at merge time in this case:

```python
# After merging segments su and sv:
nd[u] = -1; nd[v] = -1
for k in nd:
    if nd[k] == su: nd[k] = sv
# *** Check for immediate seal ***
if sv not in nd.values():
    new_ns = ns + 1          # segment sealed right now
    if new_ns > 1: skip      # multi-component → discard
```

Without this fix the DP overcounts by roughly an order of magnitude.

### State-transition rules (edge step)

At edge step for edge (u, v):

**Introduce** any vertex in {u, v} not yet in frontier: add it with label 0.

**Include / exclude decision** for edge (u, v):

| su        | sv        | Action for "include"                                      |
|-----------|-----------|-----------------------------------------------------------|
| -1 or any | -1 or any | skip (degree would exceed 2)                              |
| 0         | 0         | create new segment; assign same new ID to u and v        |
| 0         | k > 0     | u inherits label k (new endpoint); v becomes interior     |
| k > 0     | 0         | v inherits label k (new endpoint); u becomes interior     |
| j > 0     | j > 0     | cycle — discard this branch                               |
| j > 0     | k > 0     | merge segments; relabel j → k; check immediate seal       |

**Forget** vertex w when its last edge has been processed:

| sw  | Action                                                                 |
|-----|------------------------------------------------------------------------|
| 0   | vertex unvisited → discard state (not Hamiltonian)                     |
| -1  | interior vertex → remove from frontier, continue                       |
| k   | if k still appears for another frontier vertex: keep (open endpoint)   |
|     | if k no longer appears: seal (increment num_sealed); discard if > 1   |

**Final count**: after all edges processed, sum counts of states with
`num_sealed == 1` and all frontier entries equal to `-1` (or frontier empty).

---

## Improvement 3 — Compact State Encoding

Representing each state as a Python `tuple` of `(vertex, label)` pairs is convenient but
slow to hash.  A compact encoding uses a **single Python integer** (or a `bytes` object)
to pack all labels:

- Frontier size ≤ W.  
- Label values: -1 → 0, unvisited (0) → 1, segment IDs 1..k → 2, 3, …, k+1.  
  For W = 17, max segments ≈ 9, so values fit in 4 bits each.
- Total packed integer width: 4W bits (68 bits for W = 17).

Additionally, `num_sealed` (a single bit) is appended or packed into the integer.

This reduces dict-key hashing and comparison cost by roughly 10×.

---

## Improvement 4 — C Extension via cffi (for large n)

For n ≥ 60 the inner DP loop becomes the bottleneck even with compact encoding.  The
