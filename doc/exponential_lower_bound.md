# Exponential lower bound for Hamiltonian paths in the square-sum graph G(n)

## 1. Why Gerbicz's lift on its own gives only exp(N^α)

Gerbicz's ×25 step. Let b = (b_1,…,b_n) be a Hamiltonian path in G(n) with n odd, b_1 = 1 and b_n = 3. For each c ∈ [−12, 12] put

  T(c) = ( 25 b_p + (−1)^{p+1} c )_{p=1..n},   which runs from 25+c to 75+c.

Consecutive sums are 25(b_p + b_{p+1}), which are squares. T(c) holds 25b + c for b at odd positions and 25b − c for b at even positions. So T(c) ∪ T(−c) covers {25b ± c} exactly once, and the 25 paths T(c) partition [13, 25n+12]. The glue vertices 1..12 and the 25 segments are then strung into one path. The segment endpoints 25+c and 75+c do not depend on n, so a single glue arrangement works at every level. I found one from 1 to 3 (listed in §4). The output is therefore again a Hamiltonian path of G(25n+12) from 1 to 3, and the step can be repeated.

The obstruction. The paths T(c) and T(−c) must come from base paths with the same set of even-position vertices. So only the 13 classes {0}, {±1}, …, {±12} can choose their base path freely. That gives log H(25n+12) ≳ 13 log H(n), hence log H(N) ≳ N^{log 13 / log 25} ≈ N^{0.80}. The same happens with t² in place of 25: the exponent is log((t²+1)/2) / log t² < 1.

## 2. The fix: count lift-stable moves instead of paths

Definition. Let P = v_0 … v_{N−1} be a Hamiltonian path. A *move* is a window of positions [i, j] with 1 ≤ i ≤ j ≤ N−2, together with a different ordering w_i … w_j of {v_i, …, v_j} such that:
- v_{i−1} w_i … w_j v_{j+1} is a path in G, and
- w_p ≡ p (mod 2) in position, for every p. That is, each vertex keeps the parity of its position.

A family of moves is *separated* if the windows are pairwise disjoint and any two are at least one position apart.

Lemma 1. If P has k separated moves, then G has at least 2^k Hamiltonian paths with the same endpoints as P.
Proof: separated moves commute. The 2^k resulting paths are distinct because each move replaces the edge v_{i−1}v_i by v_{i−1}w_i, and w_i ≠ v_i.

Lemma 2 (lifting). Every move of the base path b lifts to a move inside each of the 25 paths T(c), and these 25k lifted moves are separated moves of the glued path P' in G(25n+12).
Proof:
- Parity is preserved, so each lifted vertex 25w ± c is unchanged and the vertex set of T(c) stays the same. The new consecutive sums are 25·(square), and the anchor sums are 25·(square) as well.
- The moves are interior to b, so the endpoints 25+c and 75+c are untouched and the glue is unaffected.
- Each T(c) is a contiguous block of P'. Whether it is traversed forwards or backwards (n odd), all positions in the block shift by the same amount, so parity is preserved in P'. Separation is clearly preserved too.

This holds for all 25 classes. A parity-preserving move does not change the even-position set, so T(c) and T(−c) can apply it independently. This is the factor of 25 per level that path counting loses.

Theorem (along a subsequence). Suppose G(n_0) (n_0 odd) has a Hamiltonian path from 1 to 3 with k separated moves. Put N_m = 25^m (n_0 + ½) − ½. Then G(N_m) has a Hamiltonian path from 1 to 3 with k·25^m separated moves, and so

  H(N_m) ≥ 2^{k 25^m} = 2^{k (2N_m + 1) / (2n_0 + 1)}.

## 3. Base certificate (computer-verified)

n_0 = 43, with the path
1 35 29 20 16 33 31 18 7 42 [39 25 24 40 9 27 37 12 13 36 28 8 41 23] 26 10 15 21 43 38 11 5 4 32 17 19 6 30 34 2 14 22 3

The move replaces
  24 | 40 9 27 37 12 | 13 36 28 8 41 | 23
with
  24 | 12 37 27 9 40 | 41 8 28 36 13 | 23

This is two odd-length (5) reversals; the chords are 24+12 = 36, 40+41 = 81 and 13+23 = 36.

Result: **H(N) ≥ 2^{(2N+1)/87} for N = 43.5·25^m − ½ = 43, 1087, 27187, …**

I checked levels 1 and 2 explicitly: N = 1087 and 27187, Hamiltonian paths from 1 to 3, with 25 and 625 separated moves.

Remarks:
- None of the 13 Hamiltonian paths of G(35) from 1 to 3 has even a parity-preserving odd reversal, so Gerbicz's own seed n_0 = 35 does not work directly.
- A random search over odd n ≤ 201 finds k ≥ 1 for many n, and k = 3 at n = 117, 125, 181. These are only random samples, not maxima.
- You don't actually need k(n_0) ≥ 1. It is enough that some level creates one new move from glue or cross edges (edges whose square is not 25·square), because every move is multiplied by 25 at all later levels.

## 4. Glue arrangement used (Hamiltonian path from 1 to 3; each pair 25+c, 75+c stands for T(c) or its reverse)

1 [15 65] [35 85] [36 86] [14 64] [80 30] [70 20] 5 11 [25 75] 6 [19 69] 12 [24 74] 7 [29 79] 2 [34 84] [37 87] [82 32] 4 [21 71] 10 [26 76] [68 18] [31 81] [63 13] [23 73] 8 [28 78] [66 16] 9 [72 22] [27 77] [67 17] [83 33] 3

(Corrected: brackets mark segments T(c), i.e. pairs {25+c, 75+c}; every other consecutive pair is a junction summing to a square, e.g. 65+35=100.)

## 5. Extending to all large N (not yet done)

Gerbicz's construction for general N uses more offsets and residues than the pure ×25 step. If each class in his general step is again a lifted base path (of size about N/25, possibly n and n±1 for different classes), then Lemma 2 applies to every class. Write ρ(N) for the ratio (number of moves)/N. Then ρ(N) ≥ ρ(n)(1 − O(1/N)) with n ≈ N/25. The product of these factors over a geometric chain of sizes converges, so ρ is bounded below uniformly. What remains for the base is a finite certificate: one path per n in a range [n_0, ~25 n_0], each with ≥ δn moves and whatever endpoint condition the general step needs.

I have not checked that Gerbicz's general step has this "every class is a lift" structure. That is the gap.

## 6. Pósa rotations

A single rotation moves an endpoint. The endpoints of every T(c) are pinned by the glue, and the global endpoints 1 and 3 are pinned for the next iteration. So rotations at endpoints only give a constant factor per class per level, and that is not multiplied at later levels. What does get multiplied is a fixed-endpoint, parity-preserving rerouting. A rotation followed by a rotation back to the original endpoint is such a rerouting (the certificate move above is of this type). So Pósa-style arguments are useful as a source of moves, within the class counted in §2.

Code (in the session workspace): lift.py, base.py, glue3.py, verify.py.
