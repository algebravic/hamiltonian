# Survey of `emathgroup/selectedTopics` for Hamiltonian-path work

Repository: <https://github.com/emathgroup/selectedTopics>. I used a shallow clone at commit `9a13e1f` (14 May 2024), which is the latest commit.

## What the repository is

It is the source of a Gatsby blog site: "selected topics" from the Chinese forum bbs.emath.ac.cn (数学研发论坛). There are 68 posts in Chinese under `content/posts/`.

The programs live in `content/attached/files/`. There are 41 `.cpp` files, 11 `.c` files and 3 `.gp` files, plus data files. Apart from `m.gp`, which is linked from `perfect-sum-ring.md`, none of the programs is linked from any post. They are the attachments of forum threads, and the explanations are in those threads, not in the repository.

No post is about the square-sum path problem. Searching the posts for A071983, A071984, A090461, Hamilton or 哈密 finds nothing. The discussion behind `path4.cpp` must therefore be on the forum itself.

## Relevant

### 1. `path4.cpp` (Zhao Hui Du): Hamiltonian paths of G_n

This file is byte-for-byte identical to our `doc/contrib/path4.cpp`. It is analysed in `doc/contrib_analysis.md`, and its pruning idea is now in our DP (`doc/sm_prune_design.md`).

### 2. `path4.2.cpp` (Zhao Hui Du, by the identical code): Hamiltonian **cycles** of G_n

This file is new to us. It differs from `path4.cpp` in three ways:

- **It counts cycles.** `init_for_N` no longer creates an initial state for each of the N(N−1)/2 endpoint pairs. Instead it creates one state per pair of neighbours {c_i, c_j} of N with c_i, c_j < N. Each state is the fragment c_i–N–c_j, meaning that vertex N has been used with its two cycle edges. The DP then starts at x = N − 1. Each undirected cycle is counted once, so the program computes **A071984**.
- **128-bit masks.** The bit mask is an `__int128_t`, which removes the N ≤ 62 limit of `path4.cpp`. N ≤ 127 still holds, because `pairs` is `char`.
- **A deeper DP stage.** D = 9 instead of 6, so the first 9 levels are merged before the search takes over. The default is `N 55`.

I compiled it with g++ 11.4 `-O2 -fopenmp` and ran it with 4 threads in the VM:

| N | cycles | Dobbelaere's A071984 value | time |
|---|---|---|---|
| 34 | 11 | 11 | 0.02 s |
| 36 | 31 | 31 | 0.09 s |
| 40 | 64 | 64 | 0.01 s |
| 42 | 1062 | 1062 | 0.06 s |
| 44 | 10091 | 10091 | 0.5 s |
| 48 | 227893 | 227893 | 10.4 s |
| 50 | 297126 | 297126 | 12.3 s |
| 52 | — | 3377189 | > 55 s (stopped) |

All the values agree.
- **Speed:** it is much faster than `path4.cpp` is for paths, because it has no O(N²) endpoint pairs. It is still slower than Dobbelaere's single-threaded backtracking, which needed about 32 s for N = 52, and far slower than our DP, which counts cycles for N ≤ 48 in well under a second.
- **Ordering:** it is the same fixed descending order as `path4.cpp`, so the same shortcomings apply (`doc/contrib_analysis.md` §2).

**What it is good for:** an independent check of A071984 by a third method, beside Dobbelaere's backtracking and our DP's `--cycles` option.

### 3. `posts/knight-tour.md` (马踏棋盘计数): counting knight's tours with a frontier DP

This post contains no code from Du; the code is KeyTo9_Fans' and is quoted in the post. It is methodologically our algorithm in a different setting.
- **The question:** in 2011, mathe asked for the number of knight's tours on the 9×10 Chinese-chess board, both closed and open.
- **The method:** KeyTo9_Fans counted closed tours with a row-by-row frontier DP, modelled as "tiling" each cell with one of C(8,2) = 28 two-port tiles. The frontier is 2 rows plus 1 cell (19 cells). Each cell records how many of its ports are fixed (0, 1 or 2), and the cells with one fixed port carry connectivity labels. This is the same state as ours: degree 0, 1 or 2 plus pairing labels.
- **Results:** closed tours on 6×n, 7×2n, 8×n (up to 8×18) and finally 9×10 = 19381952998732022416892, after two months on TB-scale disks in 2015.
- **OEIS:** A175881, A193054 and A193055.
- **Open tours** (Hamiltonian paths) on 9×10 are said to be still uncounted.

Relevance:
- **A precedent to cite.** It is a precedent for frontier DPs on graphs with bounded bandwidth, whose vertex order is fixed by the geometry, applied to Hamiltonian cycle counts.
- **A test case.** The open-tour count on 9×10 is a natural target for our DP. The knight graph on 9×10 with row-major order has a frontier of about 19 vertices, more than our 16 slots. So it would need either a better ordering (from kpw) or the 128-bit key.

The code itself is only on the forum; the links in the post point to bbs.emath.ac.cn.

## Not relevant (by topic)

These descriptions come from the post titles and, for the files, from the code itself. Most files have no comments, so their purpose is my reading of the code. I have not checked any of them.

- **Perfect difference sets** (`perfect-sum-ring.md`, `m.gp`): circular arrangements of 1..N in which every sum of consecutive beads is distinct and covers 1..N(N−1)+1 (A058241). Despite the title, this has nothing to do with square sums.
- **Pairwise square sums** (`pair-sum-square.md`): sets of 5 or 6 integers whose pairwise sums are all squares. This is a Diophantine problem, not a path problem.
- **Word games:**
  - `tse8.cpp`, `tt5.cpp`, `tte.cpp` and `t5.cpp` read `w5.txt` with 12972 five-letter words, which is the Wordle guess list. They build decision tables.
  - `bt.cpp` and `bt2.cpp` fill a grid with words from `sid.sort`, crossword style.
  - `f4.cpp` and `wf.cpp` process word lists.
- **Row-by-row transfer-matrix search on grids** (`cr2.cpp`, `cr3.cpp`): 3^W states per row of width W = 11 or 14, maximising black cells under a neighbourhood condition. It is the same transfer-matrix style, for an optimisation rather than a count.
- **Number-theoretic searches with GMP:** `abc.cpp`, `cn4.cpp`, `gc2.cpp`, `gc3.cpp`, `gmpsrch.cpp`, `mindg3.cpp`, `mnf*.cpp`, `nexterm.cpp`, `tracenext.cpp`, `newdef.cpp`, `sumi*.c*`, `se.cpp`.
- **Other enumerations:**
  - `sdv3.cpp`: subset sums of 1..N with maps.
  - `mr*.cpp`, `statc.cpp`: product-constrained arrays.
  - `vt/vt*.c`: dissections into squares.
  - `finitetree*.cpp`: curves mod p.
  - `enumps*.cpp`, `filt3.cpp`, `scfgd.cpp`, `solve8s.cpp`: configurations and equation systems, for example the "orchard planting" post.
  - `nmv2.cpp`: a 3×3 path or tiling search.

## Suggested follow-ups

1. **Cite** `path4.2.cpp` together with `path4.cpp` in the bibliography entry, or as a second entry. It is the cycle counter and agrees with A071984.
2. **Knight's tours as a benchmark:** count open knight's tours on small boards and compare with OEIS where values exist. Then estimate whether 9×10 open tours are within reach with a kpw ordering. This would be a new result if it works. The first step is to compute the pathwidth of the 9×10 knight graph.
