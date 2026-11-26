# The two contributed programs in `doc/contrib`

The two files are `A071984_opt.cpp` (Bert Dobbelaere) and `path4.cpp` (Zhao Hui Du; his name is not in the file).

## Summary

Neither program is Held–Karp. Held–Karp's state is (S, v): the set of visited vertices and the current endpoint, giving Θ(2ⁿ·n) states. Neither program uses that state.

| | `A071984_opt.cpp` | `path4.cpp` |
|---|---|---|
| Counts | Hamiltonian **cycles** (A071984), not paths | Hamiltonian **paths** (A071983), undirected |
| Method | Plain backtracking with a degree prune | Frontier ("fragment-endpoint") DP for the first 7 vertices, then a backtracking DFS |
| Vertex order | Order in which the path grows: a DFS from a fixed start triple | Fixed: N, N−1, …, 1 (descending numeric) |
| Memoization | None | Only for the first D = 6 levels, plus the starting level |
| Range | 32 ≤ N ≤ 70 (hard assert) | N ≤ 62 on LP64; N is a compile-time constant |
| Parallel | No | OpenMP over the states that come out of the DP stage |

### What I checked (Linux VM, 4 threads, g++ 11.4 `-O2`)

**`path4`** matches `data/b071983.txt` at N = 15, 16, 17, 20, 23, 25, 28, 30, 32, 35, 38, 40, 42 and 44.

| N | time |
|---|---|
| 35 | 0.94 s |
| 40 | 0.96 s |
| 42 | 11.5 s |
| 44 | 131 s |

**Dobbelaere's program** reproduces its own commented values for N = 32…52 in 2 minutes, single-threaded. N = 51 took 14 s and N = 52 took 32 s.

These timings are rough: one run each, in a VM.

---

## 1. `A071984_opt.cpp`: Dobbelaere's square-loop search

### What it computes

It counts undirected Hamiltonian cycles of the square-sum graph on {1,…,N}: OEIS A071984, "square loops". It does **not** count paths.

### Method

It is a depth-first search that extends a path one vertex at a time.

**Fixing the orientation.** Each cycle is counted once by fixing a degree-2 vertex V1 together with its two neighbours. I checked the degrees directly:

| Vertex | Neighbours | Degree 2 when |
|---|---|---|
| 50 | 14, 31 (and 71 once N ≥ 71) | 50 ≤ N ≤ 70 |
| 49 | 15, 32 (and 51 once N ≥ 51) | N = 49 |
| 32 | 4, 17 (and 49 once N ≥ 49) | 32 ≤ N ≤ 48 |

Every cycle passes through V0–V1–V2, so starting the walk as V0, V1, V2 picks exactly one of its two orientations. The trick is sound.

**Pruning.** After `latest` is extended to `nexthop`, every unused neighbour of `latest` must still have at least two *active* neighbours. Active means unused, or V0, or the new endpoint, which is still active when the test runs. This is a correct necessary condition, and it is incremental: only those vertices' available degree just dropped.

**Closing.** At depth N, a cycle is counted if `latest + V0` is a square.

### Shortcomings

1. **It counts cycles, not paths.** Turning it into a path counter needs a universal pseudo-vertex. That vertex raises every degree by 1, which destroys the degree-2 symmetry trick, and it removes most of the pruning.
2. **No memoization.** The running time is proportional to the size of the search tree. In my runs that grew by about 2.2–2.3× per unit of N between N = 50 and 52. That fits the comment's results stopping at N = 61.
3. **A hard ceiling at N = 70.** No vertex has degree 2 beyond that, and the code asserts. Dividing the count by 2 would remove the need for the trick, at twice the cost.
4. **The prune is only a degree test.** There is no connectivity test, and no check that the vertices left behind by `nexthop` can still be reached. Large dead subtrees are explored before they fail.
5. **Single-threaded.** The first-level branches are independent and could easily be parallelised.
6. **The vertex "ordering" is implicit.** It is just the DFS order; nothing chooses it. That is natural for backtracking, but it means nothing in this program carries over to a DP ordering.

---

## 2. `path4.cpp` (Zhao Hui Du): frontier DP followed by DFS

### What it computes

It counts undirected Hamiltonian paths of the square-sum graph on {1,…,N} (A071983). N is fixed at compile time with `-DN=…`.

### The endpoint trick

`init_for_N` creates one starting state for **every** pair 1 ≤ i < j ≤ N. That is N(N−1)/2 states, each of weight 1, holding a single "pair" (i, j).

Read the pair as a pseudo-edge joining the two path endpoints. A Hamiltonian path from i to j corresponds exactly to a Hamiltonian cycle of G + {ij} that uses the pseudo-edge. The program counts those cycles and sums over all pairs.

### The state (`struct IS`)

- **`pairs`.** The endpoint pairs of the current path fragments. Each pair is stored with its smaller element first, and the list is sorted by that element, so the representation is canonical. Interior vertices of a fragment are not stored.
- **`mask`.** A 64-bit set of vertices below the current vertex that already have degree 2 (they are interior) but have not been processed yet.
- **Unmentioned vertices.** Any vertex below the current one that is neither in a pair nor in `mask` has degree 0.

This is a connectivity-profile (transfer-matrix) state. Each endpoint's partner is recorded explicitly, which plays the role of our key's slot labels.

### Vertex order: fixed, descending

`update_for_x(x)` runs for x = N, N−1, …. When x is processed, every edge from x to a larger vertex has already been decided. x then chooses its edges to *smaller* neighbours c = s − x with c < x:

- **x in `mask`** (degree already 2): x is dropped and the state is copied down.
- **x not yet touched** (degree 0): x must take exactly two lower neighbours cᵢ < cⱼ. x is eliminated at once; the fragment cᵢ–x–cⱼ is recorded as the pair (cᵢ, cⱼ). If cᵢ and cⱼ are already ends of fragments, those fragments are merged, or the cycle is closed.
- **x an endpoint** (degree 1): x takes exactly one more lower neighbour and leaves the frontier.

The cycle may close only when exactly one fragment remains and every vertex 1…x−1 is already in `mask` (the test `val = (1<<x) − 2` in `makenewis`).

So the ordering is **descending numeric order**, and it is fixed. The program does not search for an ordering, and there is no parameter for one.

### Pruning and canonicalisation (`is_connected`, applied to every new state)

1. **Degree test.** For each unmasked vertex below x, count its fragment edge plus its still-available edges to other unmasked vertices below x. The edge to its own partner is excluded, since taking it would close a cycle too early. If the count is 1 and more than two such vertices remain, the state is pruned. A degree-0 test exists but is commented out; the connectivity test covers it.
2. **Connectivity test.** All unmasked vertices below x, joined by fragments and available edges, must form a single component.
3. **Forced edges.** If some endpoint has exactly one available edge, that edge is committed immediately. The test then runs again on the result, recursively. This shrinks and merges states.
4. **Counting test.** It requires `cnpair + |mask below x| < x`. It is sound but weaker than it could be; see shortcoming 6.

### The hybrid: where the DP stops

`main` runs the DP for only D + 1 = 7 levels, x = N down to N−6, storing states in a `std::map`. It then hands each surviving state, weighted by its count, to the recursive `search()`. `search()` performs the same transitions but **never merges equal states**: it is a backtracking enumeration. OpenMP parallelises over the states from the DP stage.

I also tried a full DP by setting D = N, single-threaded:

| N | Full DP | Hybrid (D = 6) |
|---|---|---|
| 40 | correct, 3.3 s, 27 MB | 0.96 s |
| 42 | not done in 40 s, RSS reached 265 MB | 11.5 s |

So D = 6 looks like a deliberate tuning choice: with this ordering and a `std::map`, the full DP is worse.

### Shortcomings

1. **A poor vertex ordering.** I computed the frontier of the descending order: the vertices below x that are adjacent to some vertex ≥ x.

   | N | 30 | 40 | 42 | 46 | 50 | 58 | 62 |
   |---|---|---|---|---|---|---|---|
   | max frontier | 17 | 25 | 27 | 31 | 33 | 40 | 43 |

   That is roughly 0.7·N. Our kpw ordering at N = 58 has pathwidth 12. The frontier is not the same as the state size, because `mask` vertices carry no matching information and pruning removes many states. Even so, the gap of 40 against 12 explains why the program has to drop memoization below 7 levels.
2. **No ordering choice at all.** Descending order is the program's built-in order.
3. **The endpoint trick costs O(N²) extra.** All N(N−1)/2 endpoint pairs start in the DP, and each pseudo-edge persists until both its ends are processed. The small vertices are processed last, so pairs with small endpoints stay in the state almost to the end. That roughly multiplies the state count by the number of possible endpoint pairs still open. Treating endpoints as degree-1 vertices within the state avoids this.
4. **Mostly exponential search.** Below level N−6 nothing is merged, so the cost is proportional to the number of leaves in the search tree. It grew about 11× from N = 42 to N = 44 (11.5 s to 131 s). Extrapolating to N = 58 is meaningless: hours at the very least, almost certainly far more.
5. **Expensive per state.**
   - `std::map` costs O(log M) per insertion and roughly 50–100 B of node overhead for each `IS`, which is itself N + 16 bytes.
   - `is_connected` scans every vertex below x and every square, builds a union-find from scratch, and may recurse.
   - `mbc` is an O(x) loop where a popcount would do.
6. **A weak counting test.** Every fragment endpoint lies below x, so the true constraint is 2·cnpair + |mask| ≤ x − 1. The code tests cnpair + |mask| < x. That is correct but prunes less.
7. **Hard-wired limits.**
   - `mask` is a `long` and bits are set with `1L<<i`, so N ≤ 62 on LP64. At N = 63, `(1L<<step)` overflows (undefined behaviour). On LLP64 (Windows), `long` is 32 bits and the program fails for N ≥ 32.
   - Counts are a signed `long`, which is fine for N ≤ 62.
   - The program must be recompiled for each N, and D is a fixed `#define`.
8. **Minor code-quality points.**
   - The `IS` arrays are uninitialised beyond `2*cnpair`. This is harmless, because comparison and lookup only read the first `2*cnpair` entries.
   - The capacity check runs only after `add_new_pair` has written `pairs[2*cnpair+1]`. That is safe in practice, since cnpair ≤ frontier/2 ≈ 0.35·N, but fragile.
   - If the sorted-insertion code ever produced a non-canonical order, the result would still be correct (lookups are linear scans), but equal states would no longer merge. Only efficiency would suffer.
   - OpenMP hands out work one state at a time under a `critical` section. That is fine here, because each task is a large subtree.

### Correctness

I found no correctness bug, and the results match A071983 up to N = 44. Each step's logic is sound by the arguments above:
- each vertex receives exactly its two cycle edges;
- edges to higher vertices are already decided when a vertex is processed;
- a forced edge is the unique edge left for that endpoint, so committing it early loses no solution;
- every prune is a necessary condition.

I have not proved the pair-list update routines correct line by line. Their agreement with the reference data is evidence, not proof.

---

## 3. How they relate to our sort-merge DP

- **`path4`** has essentially our state model: fragment endpoints with partner labels, plus "already degree 2" information, in a frontier DP. It differs in three ways:
  - it uses a fixed, wide ordering (frontier about 0.7·N, against pw 12–16 for kpw);
  - it switches to search after 7 levels instead of merging states throughout;
  - it uses a `std::map` rather than sort-merge tables.

  Its useful idea is the forced-edge and connectivity pruning, which our DP did not do. It has since been tested in our DP: see `doc/sm_prune_design.md`.
- **Dobbelaere** is a pure search for cycles. Its degree prune is the same idea as path4's degree test. It contributes no ordering or DP technique, but it is a useful independent check on A071984 for N ≤ 61.
