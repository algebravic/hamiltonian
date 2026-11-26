The question
============
Define the undirected graph G(n) for each positive integer n as
follows: G(n) has n vertices labeled 1 through n. There is an edge
between i and j if and only if i+j is a perfect square. Explicit
calculations for n<=75 show that G(n) has at least an exponential in n
Hamiltonian paths. I would like to prove an exponential lower bound
for sufficiently large n.

I know that Gerbicz has an explicit inductive construction which shows
the existence of a Hamiltonian path for all sufficiently large n, but
I don't see how to use that to get an exponential lower bound.

Can we use Posa rotations along with Gerbicz's construction?

Answer
======
Quick finding: none of the 13 base paths 1→3 in G(35) has a parity-preserving reversal. Next I'll check the glue step and look at larger bases.

Glue arrangement 1→3 exists, so the ×25 lift can be iterated. Now I'll check the whole chain end to end for a base with one move.

I have a proof of an exponential lower bound along the subsequence N_m = 43.5·25^m − ½ (43, 1087, 27187, …). It does not yet cover every large N; the remaining gap is in Gerbicz's general step.

**Why the ×25 lift alone isn't enough.** Gerbicz's step takes a Hamiltonian path b of G(n) from 1 to 3 (n odd) and builds 25 paths, one for each c in −12..12:

T(c) = (25b_p + (−1)^{p+1} c), running from 25+c to 75+c.

These paths partition [13, 25n+12], and the glue vertices 1..12 string them together. The segment endpoints don't depend on n, so a single glue arrangement works at every level. I found one that runs from 1 to 3, so the step can be repeated. The catch: T(c) and T(−c) must come from base paths with the same set of even-position vertices. So only 13 of the 25 classes can choose their base path freely. That gives log H(25n+12) ≳ 13·log H(n), which grows only like N^{log 13/log 25} ≈ N^{0.80}.

**The fix: count moves, not paths.** Define a *move* as an interior window of the path that can be re-ordered so that:
- the endpoints of the window stay fixed,
- the result is still a path in G, and
- every vertex keeps the parity of its position.

Such a move doesn't change the even-position set. So it lifts to a valid move inside every T(c), and T(c) and T(−c) can apply it independently. Since n is odd, all 25 lifted copies are again parity-preserving moves of the glued path. Moves in disjoint windows with at least one vertex between them can be applied independently. So:

- k separated moves give at least 2^k Hamiltonian paths;
- k(25n+12) ≥ 25·k(n), which grows linearly in N.

**Base case (computer-checked).** G(43) has a path from 1 to 3 with one move: two odd-length reversals. The block 24 | 40 9 27 37 12 | 13 36 28 8 41 | 23 becomes 24 | 12 37 27 9 40 | 41 8 28 36 13 | 23. This gives

**H(N_m) ≥ 2^{(2N_m+1)/87}.**

I built the paths at levels 1 and 2 explicitly (N = 1087 and 27187). Each is a Hamiltonian path from 1 to 3, with 25 and 625 separated moves respectively, all verified. Gerbicz's own seed n = 35 doesn't work: none of the 13 paths of G(35) from 1 to 3 has such a move. You don't actually need a move in the base, though. One new move created by the glue at any level is enough, because every later level multiplies it by 25.

**Pósa rotations.** A single rotation moves an endpoint. Here the endpoints are pinned: the glue fixes the ends of every T(c), and the next iteration needs the path to run from 1 to 3. So endpoint rotations give at most a constant factor per class per level, and that factor is not multiplied at later levels. What gets multiplied is a re-ordering with fixed endpoints that preserves position parity. A rotation followed by a rotation back to the original endpoint is exactly that (the G(43) move is of this type). So Pósa-style arguments are useful as a way to find moves, as long as you stay within that class.

**What's left for all large N.** Gerbicz's construction for general N uses other offsets than the pure ×25 step. If in his general step every class is still a lift of a base path of size about N/25 (sizes n and n±1 are fine), the move count still grows by a factor of 25 per level. The losses per level are a factor of (1 − O(1/N)), and their product over geometrically growing N converges, so the moves stay at least δN for every N. The base would then be a finite computation: one certificate path per n in a range of about [n₀, 25n₀]. I have not checked that his general step has this "every class is a lift" structure; that is the gap.

Files are in `exponential_lower_bound.md`, `verify.py`, `base.py` and
`lift.py`; the glue arrangement is listed in §4 of the write-up.

Question
========
Can you explain why the factor of 25 arises in Gerbicz's construction?
Is it that the factor needs to be a square and smaller squares don't
work, or is there some other strong structural reason?

Answer
======
It's partly the square requirement, but the specific choice of 25 is not forced by deep structure. 25 is the smallest odd square for which the finite gluing problem can be solved. The square requirement rules out every non-square factor, odd t follows from how the residues pair up, and a computation plus a spacing heuristic rule out 9.

**1. Why the factor must be a square.** The lift only works if multiplying by m sends edges to edges, which means m·k² must be a square for every k. That forces m = t². The additive offsets then have to alternate in sign along the path, x_p = t²·b_p ± c, so that they cancel in every consecutive sum. That is why the offsets come in ±c pairs.

**2. Why t should be odd.** The 2h+1 lifted paths, one per offset c ∈ [−h, h] with h = (t²−1)/2, partition [h+1, t²n+h]. The leftover numbers 1..h are the glue. This uses one representative of each residue class mod t², and that only works cleanly when t² is odd. When t is even, the class of t²/2 is its own negative, so the lift with offset c = t²/2 has nothing to pair with. Covering each number t²(k+½) exactly once would then require every vertex of the base path to sit at a position of the same parity, which is impossible.

**3. Why the endpoints are 1 and 3.** For the step to repeat, the output must again run from u to w. Since the output starts and ends on glue numbers, u and w must lie in 1..h. The lifted paths then have endpoints near t²u and t²w, which don't depend on n. So the gluing is a fixed finite Hamiltonian-path problem: string together h glue numbers and 2h+1 forced endpoint pairs, using square sums. Each value of t succeeds or fails once and for all.

**4. Why not 9.** I checked t = 3 exhaustively in this scheme. I tried every endpoint pair u ≠ w in {1,…,4}, with n both odd and even (the two sign patterns at the ends), and there is no glue arrangement in any of the 24 cases. For t = 5, a randomized search found arrangements for (1,3) and (1,2) with both parities. For t = 7 it found one for (1,2) with n even; the other t = 7 cases I tried were inconclusive within the search budget, not ruled out.

**5. A heuristic for why 5 is the threshold.** The lifted endpoints sit in windows of width about t² around t²u and t²w. Most junctions in the glue path join two such windows, or a window and a glue number, with sums around t²(u+w). Near such a sum, squares are spaced about 2t√(u+w) apart. The windows reliably contain enough square-sum pairs only when t² is larger than that spacing, i.e. t > 2√(u+w). With (u, w) = (1, 3) that means t > 4. For t = 3, even the smallest choice u+w = 3 gives 2√3 ≈ 3.46 > 3. So at t = 3 there are too few square sums available to make the 12 junctions; at t = 5 there are plenty.

So 25 is the first value where this small finite problem has a solution, and t = 5 also keeps the problem small. The t = 3 result only covers this exact scheme: glue set exactly {1..h} and base endpoints in {1..h}. A variant with other glue sets or larger endpoints might work with 9; I haven't excluded that.

This bears on your exponential bound. The moves in the base path get
multiplied by t² per level, and N grows by the same t², so the density
of moves is preserved for any odd t ≥ 5 whose gluing closes. A larger
t buys nothing per level, but it gives more subsequences. That could
help in covering all N, if you combine several values of t with
different seeds.

Question
========
Here's something extremely speculative: since the graph G(n) is an
induced subgraph of G(n+1) it makes sense to define the direct limit
G(infinity). I know that graphons can be defined for dense families of
graphs, but this family isn't dense. So, is there some sort of
modification to the definition of graphon that can apply here?

Answer
======
Yes, with one twist. The sparse-graphon theories apply in principle, but the natural limit for G(n) is not a function on [0,1]². It lives on [0,1] × Ẑ, where Ẑ is the profinite integers, and on the Ẑ factor it is a singular measure. Working that out gave a concrete prediction for log H(n), which matches your counts unexpectedly well.

**1. Which regime.** Vertex i has about √(n+i) − √i neighbours, so every degree is Θ(√n). The edge density is p_n ≍ n^{-1/2}. That is too sparse for ordinary graphons and too dense for Benjamini–Schramm local limits. It is the intermediate regime that the L^p graphon theory of Borgs–Chayes–Cohn–Zhao, "measures on the square" (Kunszenti-Kovács–Lovász–Szegedy) and action convergence (Backhausz–Szegedy) were built for.

**2. The obvious candidate fails.** Rescale the adjacency step function by √n. Near a sum m the squares have density 1/(2√m), so the averaged limit is

  κ(x,y) = 1/(2√(x+y)),

which is in L^p for p < 4. But G(n) does not converge to κ in cut norm. Take S = T = the multiples of 4. Their sums are ≡ 0 mod 4, and a number ≡ 0 mod 4 is a square twice as often as a typical number (k² ≡ 0 mod 4 has the two solutions k ≡ 0, 2). That gives a discrepancy of fixed size for every n.

It gets worse. Partition the vertices by residue mod Q, with Q a primorial. At each odd prime ℓ, half the residues are nonzero squares, and each has 2 square roots. So the L^p norm of the block-averaged rescaled graph grows like 2^{(p−1)ω(Q)}, which is unbounded. The sequence is therefore not L^p upper-regular for any p > 1, and the Borgs–Chayes–Cohn–Zhao theory does not apply as it stands. I have done this calculation only at heuristic level.

**3. A limit that encodes the arithmetic.** Send vertex i to the pair (i/n, i) in Ω = [0,1] × Ẑ, with Lebesgue × Haar measure on Ω. The natural limit is the symmetric measure on Ω² given by

  κ(x,y) dx dy ⊗ ν(dα, dβ).

Here ν is the law of (α, σ − α), where α is Haar-distributed and σ = k² for a Haar-random k. In other words, the pair sums to a "random profinite square". Squares form a Haar-null set in Ẑ, so ν is singular with respect to Haar ⊗ Haar. The limit is therefore a measure-valued kernel, which is exactly the kind of object in the measures-on-the-square and action-convergence frameworks.

Convergence against test sets of the form (interval) × (residue class mod q) follows from the equidistribution of squares in residue classes. I expect, but have not proved, that the full sequence G(n), with adjacency scaled by √n, converges in action convergence to this operator.

On G(∞) itself: since the G(n) are initial segments, this is really a scaling limit of G(∞) viewed at scale n. The archimedean part is scale-covariant, κ(nx, ny) = n^{-1/2} κ(x,y), which is graphex-like, but G(∞) is not exchangeable.

**4. The payoff: a sharp prediction for H(n).** Use the limit in a first-moment (Sanov / Schrödinger-bridge) heuristic. Then

  log H(n) ≈ log(n!/2) − ((n−1)/2) log n + C·n + o(n),

  C = sup over couplings π of (uniform, uniform) of [ ∫ log κ dπ − KL(π ‖ λ²) ] = −2 ∫₀¹ log a,

where a solves a(x) ∫₀¹ a(y) κ(x,y) dy = 1. Symmetric Sinkhorn iteration gives C ≈ −0.63254, stable under grid refinement.

The profinite factor contributes nothing at exponential order. At each prime the local kernel is translation-invariant, ρ(i+j), so π ∝ ρ is already a coupling and attains log ∫ρ = 0.

Here are the residuals, log H(n) − prediction, against the counts I had to hand:

| n | 35 | 40 | 45 | 57 | 59 | 61 | 62 | 63 |
|---|---|---|---|---|---|---|---|---|
| residual | 0.89 | −1.81 | 0.10 | −1.50 | −1.74 | −1.19 | −0.50 | 0.40 |

All residuals are O(1) while log H(n) itself is about 32 at n = 63. These are only eight points. There could be a hidden α·log n term, and nothing in the heuristic makes it rigorous for a deterministic graph. Checking for drift in the residual through your n ≤ 75 data would be the obvious test.

If the prediction holds, H(n) grows like (√n/e)^n e^{Cn}, far beyond exponential.

**5. A possible rigorous route that connects to your Pósa question.** In G(n,p), the number of Hamilton cycles is n!pn(1+ o(1))n asymptotically almost surely (a.a.s.), provided that p ≥ ln n+ln ln n+ω(1)/n (Glebov–Krivelevich). The pseudo-random version of the argument runs in two steps. 1. We first use (6) and the Egorychev-Falikman theorem to estimate from below the number of 2-factors in G, weighted by their numbers of cycles. 2. Then we prove that the contribution of 2-factors with many cycles to this number is rather insignificant. The cycles of a factor are then merged by rotation–extension.

For G(n), Egorychev–Falikman applied to the Sinkhorn-scaled adjacency
matrix should give, rigorously, about (√n/e)^n e^{Cn} cycle covers,
provided the discrete scaling vectors track a(x). The hard part is the
merging step: it needs expansion for small sets, and the arithmetic
structure (mod 4, and so on) may get in the way. That merging is where
Pósa rotations would do real work, and if it goes through you would
get a superexponential bound, not just an exponential one.

Question
========
This is quite interesting. Can you write this up as a latex paper
along with precise references to the literature? All of this should be
attached to the repo in ~/Programming/Combinatorics/hamiltonian. The
latex file should go in the doc subdirectory, and any python programs
in the hamiltonian/bounds subdirectory.

Answer
======
Writing the code first. One claim from earlier needs checking before it goes in the paper: I only tested G(35) for odd reversals, not for general parity-preserving moves.

Seed n₀ = 45 has two separated moves, which improves the constant to 2^{(4N+2)/91}. Now the plan file and the paper.

Quick check that it runs in place in the repo:

I've written the paper, `doc/square_sum_bounds.tex` (8 pages, compiled to the PDF alongside it), and put the code in `hamiltonian/bounds/`. The certificate check runs in the repo and passes. Nothing is committed to git.

**What's proved and what isn't.** The paper marks this throughout:
- **Theorems:**
  - the lifting step and the lemma that parity-preserving moves lift into all 25 classes;
  - the exponential bound along the subsequences;
  - the exact edge count of G(n);
  - the mod-4 fact that rules out the obvious limit kernel.
- **Computer-verified:**
  - the seed paths;
  - no glue arrangement exists for t = 3 (exhaustive);
  - the explicit lifts to N = 1137 and 28437.
- **Conditional or conjectural:**
  - extending the bound to all large N, which needs Gerbicz's general step to have the "every class is a lift" form;
  - the profinite limit;
  - the asymptotic constant C.

**Two corrections to what I told you earlier:**
- **Gerbicz's own seed, G(35), works after all.** It has no parity-preserving moves in short windows, but it does have them in long ones (window length 15 or more). That gives H(N) ≥ 2^{(2N+1)/71} along N = (71·25^m − 1)/2.
- **There is a better seed: G(45) has a path from 1 to 3 with two separated moves.** So the main theorem is now **H(N) ≥ 2^{(4N+2)/91}** for N = (91·25^m − 1)/2. Exhaustive search shows no seed below 41 has two moves. Random sampling up to 69 found nothing better than 45.

**Fit to your counts.** Using all of `data/b071983.txt`, the residuals from the conjectured asymptotic stay in [−2.68, 1.24] for every n from 15 to 74. The least-squares fit suggests an extra −log n term, but the range is too short to say.

**Please recheck H(75).** H(75)/H(74) is about 300, while the neighbouring ratios are 2 to 5, and its residual jumps by 4.7. Vertex 75 has only four neighbours (6, 25, 46, 69), so this looks like a possible error in the count.

**References.** I checked these against the publisher or arXiv pages:
- Borgs–Chayes–Cohn–Zhao I and II
- Kunszenti-Kovács–Lovász–Szegedy
- Backhausz–Szegedy
- Glebov–Krivelevich
- Krivelevich (pseudo-random graphs)
- Kenyon–Král'–Radin–Winkler
- Jooken–Zamfirescu
- Berlekamp–Guy

The classical ones (Egorychev, Falikman, Bregman, Pósa, Sinkhorn, Cooper–Frieze, Cuckler–Kahn, Léonard, Borgs–Chayes–Cohn–Holden) are from my own knowledge and worth a quick proofread. I cite Ferber–Kwan–Sudakov and Veitch–Roy by arXiv number only, since I couldn't confirm their journal versions.

**Code.** All modules use relative imports; run them as `uv run python -m hamiltonian.bounds.<module>`. Before writing them I put the plan in `doc/bounds_plan.md`.
- `certificate.py`: checks the seeds and the lifts.
- `glue.py`: glue search; `--t 3 --exhaustive` reproduces the "none for 9" result.
- `seeds.py`: searches for seed paths.
- `sinkhorn.py`: computes C.
- `residuals.py`: produces the table of residuals.
- `edges.py`: checks the edge count and the mod-4 factor.

The seed and glue data are in `data/bounds/certificate.json`. I overwrote your copy of `lift.py`. Your older `base.py` and `verify.py` (and the `~` backups) are still there, because deleting files isn't enabled; they are superseded and can be removed.

Files are in ~/Programming/Combinatorics/hamiltonian:
- doc/square_sum_bounds.tex
- doc/square_sum_bounds.pdf
- doc/bounds_plan.md
- data/bounds/certificate.json
- hamiltonian/bounds/

Question
========
I can see you point about n=75. That was an especially difficult
calculation, as I had to keep adjusting the parameters in machine.yaml
in order for it not to crash. It was run on a machine with 1 TB memory
and 64 processors. I don't have access to it anymore. I would like to
give a talk about this, but the audience is probably not familiar with
graphons (or graphops or graphexes). I'd like to structure it so as to
give them a reasonable idea of what the conjectured approach is in the
last section of the paper.

Answer
======
Beamer talk, about 50 minutes, aimed at a research audience. While planning it I found a rigorous upper bound that makes the last section much stronger, so I'll check it first.

Now I'll add the upper bound and the mod-4 structure to the paper:

Now the talk itself:

The talk is in `doc/talk/` as a beamer file with 24 frames, sized for about 50 minutes. While building it I also found a rigorous upper bound that makes the last part of the talk much stronger.

**New result: a proved upper bound.** I added a vertex joined to every other vertex. Each directed Hamiltonian path then closes up into a cycle cover of the enlarged graph. Bregman's permanent bound, applied to the degree profile √n·(√(1+x) − √x), then gives

  log H(n) ≤ (n/2) log n + (B−1)n + o(n),  B = √2/2 − (3/2)·log(1+√2) = −0.614954…

The conjectured constant C = −0.632538 is strictly smaller, by the Gibbs variational principle, and sits only 0.0175 below B. So the conjecture says the truth is very close to Bregman, and the open problem is entirely the lower bound. I added this to the paper as a new proposition, with the proof.

**How the talk is built for people who haven't seen graphons**
- **Part I (lifting):** the ×25 lift and the moves argument. No graph-limit background is needed.
- **Part II (graph limits):** a graphon is introduced as the limit of the adjacency matrix drawn as a pixel picture. Then the actual picture of G(n): edges lie on the anti-diagonals i + j = k², which average out to κ(x,y) = 1/(2√(x+y)). Sorting the vertices by residue mod 4 shows half the blocks empty and the other half at twice the density. That is why the limit has to remember residues. The formal frameworks (graphops and the rest) get one line, only so people know where to look.
- **Part III (counting):** first-moment counting framed as energy versus entropy. Hamiltonian cycles would like to use pairs of small numbers, but must use every vertex once. Then the Schrödinger/Sinkhorn solution, the plot of the data against C, and the Bregman bound.
- **Part IV (route to a proof):** the template is van der Waerden for many cycle covers, then few cycles, then merging by Pósa rotations. The slides say exactly what would have to be proved (Lemmas L1–L3) and where the arithmetic gets in the way.

**Two things to know before presenting**
- **G(n) is not pseudo-random.** The residue classes 1 and 3 mod 4 are independent sets. So is the class 1 mod 3, of size about n/3. There are no edges between the classes 0 and 2 mod 4. Standard rotation arguments assume any two large sets are joined by an edge, so they need a residue-aware form. I've noted that the quotient graphs on residue classes are connected, which is what makes this plausible.
- **Lemmas L1–L3 are not proved and are labelled that way.** Every slide carries a tag: [proved], [computer-verified], [heuristic] or [conjecture].

**On H(75):** since you can't rerun the exact computation, a cheap check is to run your importance-sampling estimator (`ham_is.py`) at n = 74 and n = 75. It only needs to tell a ratio of about 300 from a ratio of about 3 to 5, and even a rough estimate can do that.

The other new code is `hamiltonian/bounds/bregman.py` (the finite and asymptotic bound) and `hamiltonian/bounds/talk_figures.py`, which writes the figures into `doc/talk/figs/`. I described both in `doc/bounds_plan.md` before adding them.

Files are in ~/Programming/Combinatorics/hamiltonian:
- doc/talk/square_sum_talk.tex
- doc/talk/square_sum_talk.pdf
- doc/talk/talk_notes.md
- doc/talk/figs/
- doc/square_sum_bounds.tex
- doc/square_sum_bounds.pdf
- doc/bounds_plan.md
- hamiltonian/bounds/bregman.py
- hamiltonian/bounds/talk_figures.py

Question
========
On slide 3 you don't say what N is (as opposed to n). I'm assuming
that it means a subsequence from Gerbicz. Is that right? Also, it
would be nice to have an illustration of gluing two sequences in
Gerbicz's construction.

Answer
======
Yes, N is a subsequence, but not quite Gerbicz's. It is N = (91·25^m − 1)/2 = 45, 1137, 28437, …, the iterated lifts of a seed path in G(45). Gerbicz's own seed, G(35), gives his sequence (71·25^m − 1)/2 with the weaker bound 2^{(2N+1)/71}. Slide 3 should say all this; I'll fix it.

Drawing the gluing exposed a real error, which I'll also fix. The glue table in the paper (and in the markdown write-up I gave you earlier) brackets the wrong pairs. In the list `1 15 65 35 85 …`, the segment T₋₁₀ is 15→65, and 65+35 = 100 is a *junction* between two segments; the table bracketed the junction pairs instead of the segments. The code always parsed it correctly, which is why the verification passed. I've checked all 36 junction sums in the corrected version.

Now fixing the paper's glue table and adding the same figure there:

Good. The older `doc/exponential_lower_bound.md` in your repo has the same wrong bracketing, so I'll fix it in place:

The new slide shifts the numbers in the speaker notes, so I'll update those too:

Slide 3 now defines N, and slide 5 is a new gluing illustration. Drawing the gluing also exposed a wrong glue table in the paper, which I've corrected.

**What N is.** As I said above, it's the sequence N = (91·25^m − 1)/2 = 45, 1137, 28437, …, built from the G(45) seed; Gerbicz's own seed gives his sequence (71·25^m − 1)/2 and the weaker bound. Slide 3 now says this.

**The gluing slide.** It uses the actual seed path b = (1, 35, 29, 20, …, 33, 3) in G(45).
- **Top row:** the whole glued path in outline. It goes from glue vertex 1, through T₋₁₀ (15⋯65), T₁₀ (35⋯85), T₁₁ (36⋯86), …, to T₈ traversed backwards, and ends at glue vertex 3. Each junction is labelled with its square: 16, 100, 121, …, 36.
- **Below:** T₋₁₀ = 15, 885, 715, 510, …, 835, 65 and T₁₀ = 35, 865, 735, 490, …, 815, 85, drawn as two rows. Inside each segment the sums are 900, 1600, 1225, …. These are 25 × (1+35), 25 × (35+29), 25 × (29+20), copied from b.
- **The junction:** the end of the first row (65) is joined to the start of the second (35) by an ordinary square, 65 + 35 = 100, which is not a multiple of 25.

That is the whole idea of the construction: square sums inside a segment are inherited from b, and the joins between segments are a small square-sum puzzle that is the same at every level.

**The wrong table.** The glue table in the paper, and in the earlier `doc/exponential_lower_bound.md`, bracketed the junctions instead of the segments. The first segment is [15 65], not [65 35]. The code always read the list correctly, so all the verified results stand. I've corrected both files and checked all 36 junction sums, and the paper now includes the same gluing figure.

The figure source is `doc/figs/gluing.tex`, which both the paper and the talk read in. I also updated the slide numbers in `doc/talk/talk_notes.md`, since the new slide shifts everything after slide 4 by one.

Files are in ~/Programming/Combinatorics/hamiltonian:
- doc/talk/square_sum_talk.tex
- doc/talk/square_sum_talk.pdf
- doc/talk/talk_notes.md
- doc/figs/gluing.tex
- doc/square_sum_bounds.tex
- doc/square_sum_bounds.pdf
- doc/exponential_lower_bound.md

Question
========
On slide 7 you first talk about k. There it's just a particular
parameter. But the lemma on slide 8 talks about the function k(n). Are
you saying that the choice of k depends only on n and not on the
individual sequence. It's not clear?

Answer
======
You're right, the slides blur two things. The count of moves is a property of an individual path. k(n) should be defined as the best count over all suitable paths of G(n). The lemma is really about a path, and the inequality for k(n) follows from it. I'll make slides 7 and 8 say exactly that.

No, k doesn't depend only on n. The number of moves is a property of each individual path, and different Hamiltonian paths of the same G(n) can have different numbers. The slides blurred this, and I've fixed them:

- **Slide 7:** now says the move count belongs to a path P. If P has k separated moves, applying any subset of them gives 2^k distinct Hamiltonian paths with the same ends as P.
- **Slide 8, the lemma:** now stated for a single path. If a base path b in G(n) has k separated moves, its lift P′ in G(25n+12) has 25k.
- **Slide 8, the consequence:** for odd n it defines k(n) as the largest number of separated moves over all Hamiltonian paths of G(n) from 1 to 3. Applying the lemma to a path that achieves this maximum gives k(25n+12) ≥ 25·k(n).
- **Slide 8, the theorem:** now says *some* path of G(45) from 1 to 3 has two separated moves, i.e. k(45) ≥ 2.

The proof never needs the maximum. It only needs one path with k moves, and then follows the lifts of that one path. k(n) is there so the growth claim can be stated as a function of n.

The paper already states the lemma and theorem per path, and the definition of k(N) in its "All large N" subsection is also a maximum over paths, so it needed no change.

The updated slides are in ~/Programming/Combinatorics/hamiltonian:
- doc/talk/square_sum_talk.tex
- doc/talk/square_sum_talk.pdf

Question
========
I don't need it on the slides but do you have a reference for the use
of Schroedinger bridge and the potential along with Sinkhorn
scaling. So I can it worked out in other cases in detail?

Answer
======
The key step is recognising that our constant is a static Schrödinger bridge in disguise. Take the reference measure R(dx dy) = κ(x,y) dx dy / Z with Z = ∫∫κ. Then

  C = log Z − min over couplings μ of (λ, λ) of KL(μ ‖ R).

In the entropic optimal transport literature this is the case cost c = −log κ, regularisation ε = 1, and uniform marginals. The minimiser has the form μ = e^{φ(x)+ψ(y)} R. The functions φ and ψ are called the *Schrödinger potentials*; by symmetry here e^{φ} = e^{ψ} = √Z · a. Sinkhorn's algorithm (also called IPFP) is alternating projection onto the two marginal constraints. With that dictionary, here is where to read up.

**Theory of the potentials and of Sinkhorn**
- **Nutz, *Introduction to Entropic Optimal Transport*** (lecture notes, Columbia, version of Dec. 2022). The cleanest self-contained account. It develops mathematical foundations for entropic optimal transport and Sinkhorn's algorithm in a self-contained yet general way, with chapters on static Schrödinger bridges, duality, and Sinkhorn's algorithm. It uses exactly the reformulation above: inf over π∈Π(µ,ν) of H(π|R), by introducing the auxiliary reference measure R with dR ∝ e^{−c/ε} d(µ⊗ν). Available at math.columbia.edu/~mnutz/docs/EOT_lecture_notes.pdf.
- **Borwein, Lewis and Nussbaum, "Entropy minimization, DAD problems, and doubly stochastic kernels"**, J. Funct. Anal. 123 (1994), 264–307. This is the continuous Sinkhorn problem itself. The classical DAD problem asks, for a square matrix A with nonnegative entries, when it is possible to find positive diagonal matrices D1 and D2 with D1AD2 doubly stochastic, and the paper treats various continuous and measurable generalizations of this problem. This is the reference for existence and uniqueness of a(x).
- **Nussbaum, "Entropy minimization, Hilbert's projective metric, and scaling integral kernels"**, J. Funct. Anal. 115 (1993), 45–99 . This proves convergence of the iteration as a contraction in Hilbert's projective metric.
- **Two surveys:** Idel's review of matrix scaling (arXiv:1609.06349) covers over 70 years of matrix scaling. Léonard's survey of the Schrödinger problem (DCDS-A 34 (2014)) is already cited in the paper.
- **Peyré and Cuturi, *Computational Optimal Transport***, Found. Trends Mach. Learn. 11 (2019); arXiv:1803.00567. Chapter 4 covers the numerics. This one is from my own knowledge, so check the details.

**The same variational principle used for counting permutations (closest to our use)**
- **Mukherjee, "Estimation in exponential families on permutations"**, Ann. Statist. 44 (2016), 853–875. He computes asymptotics of the normalizing constant for a class of one parameter exponential families on permutations which includes Mallows model with Spearmans's Footrule and Spearman's Rank Correlation Statistic, and an iterative algorithm (IPFP) is proved to converge to the limiting normalizing constant. His normalising constant Σ_π exp(θ Σ_i f(i/n, π(i)/n)) is exactly our E_π ∏κ, with f = log κ and θ = 1. It has worked examples: f = |x−y| (footrule) and f = xy (rank correlation). His theorems assume a continuous f, so our log κ, which tends to −∞… correction: log κ is unbounded *above* at the corner (0,0), so it needs an extra truncation step.
- **Mukherjee and Tagami** (arXiv:2304.04074) make the link explicit: the proposed model in [28, Eq (1.1)] has connections to entropic optimal transport (EOT).
- **Starr, "Thermodynamic limit for the Mallows model on S_n"**, J. Math. Phys. 50 (2009) (arXiv:0904.0696). This works out the limiting density in closed form for the Mallows model. That model has a two-body weight (inversions) rather than our one-body weight, so it is a fully worked example of the method rather than of our exact equation.

**Connecting permanents to scaling (Step 1 of the paper)**
- **Bochi, Iommi and Ponce, "The Scaling Mean and a Law of Large Permanents"** (arXiv:1502.02997). It relates the permanental mean and the scaling mean, using results including the van der Waerden conjecture. This is the closest thing I found to Lemma L1. It doesn't treat a kernel sampled on a grid directly, but its "scaling mean" is the same object as our ∏ d_i.

**Caveat for applying these to κ.** Most of these results assume a kernel bounded above and below. Our κ is bounded below on [0,1]² (κ ≥ 1/(2√2)) but has an integrable singularity at (0,0). So the Hilbert-metric contraction argument doesn't apply as stated. The measurable-kernel results of Borwein–Lewis–Nussbaum are the place to check existence. The numerics were stable: C agreed to 6 digits across grids of 400, 1600 and 3200 points.

If you'd like, I can add the relevant ones to the paper's
bibliography, next to the Schrödinger-bridge discussion in §5.

Question
========
Yes, add them to the paper bibliography.

Answer
======
I've added the nine references to the paper. They're grouped by purpose in a new Remark 5.2, right after the Sinkhorn computation of C.

- **The dictionary:** the remark first states C = log Z − min_{μ∈Π(λ,λ)} KL(μ‖R), with R = κ dx dy / Z. This is the static Schrödinger bridge with cost −log κ, regularisation ε = 1 and uniform marginals, in Nutz's notation. The potentials are e^φ = e^ψ = √Z · a.
- **Existence and uniqueness of the continuous scaling:** Borwein–Lewis–Nussbaum (1994) and Nussbaum (1993).
- **Surveys:** Idel for matrix scaling; Léonard and Nutz for Schrödinger bridges.
- **Numerics:** Peyré–Cuturi.
- **The same formula for exponential families on permutations, with worked examples:** Mukherjee (Ann. Statist. 2016) and Mukherjee–Tagami.
- **A fully worked example with a two-body weight (the Mallows model):** Starr (2009).
- **Permanental means versus scaling means**, relevant to Lemma L1: Bochi–Iommi–Ponce.

The remark ends with the caveat about the singularity of κ at (0,0). The contraction argument of Nussbaum (1993) doesn't apply verbatim, and Mukherjee's setting needs log κ truncated near the corner.

I also sorted the whole bibliography alphabetically, since the new entries had broken the order.

Two of the new references rely on my own knowledge rather than a check against the source:
- **Peyré–Cuturi:** I gave Found. Trends Mach. Learn. 11 (2019), 355–607; please verify the volume and pages.
- **Starr:** I couldn't confirm the article number in J. Math. Phys. 50 (2009), so the entry gives only the volume, year and arXiv number.

All the others match their publisher or arXiv pages.

The updated paper is in ~/Programming/Combinatorics/hamiltonian:
- doc/square_sum_bounds.tex
- doc/square_sum_bounds.pdf

Question
========
I find the conjecture for the asymptotics of the number of Hamiltonian
paths in G(n) quite compelling, However, the only problem that I see,
is that the asymptotics are only whp (probably a.s.) for a finite
graph sampled from the graphop. Is there a path to applying it to the
specific family G(n)?

Answer
======
Before answering, I'll check numerically the step of the proof route that already has a deterministic tool.

Yes, there's a path. The random model R(n) was only used to guess the constant C, and the proof template it points to can be run deterministically on G(n). In the deterministic version (Krivelevich's pseudo-random-graph argument), randomness only ever supplies a few structural inputs. For G(n) each of those inputs becomes a concrete statement about squares, so the conjecture is really about the deterministic family; the graphop only predicts the answer.

**Step 1 (counting cycle covers) is already deterministic, and the numbers match C.** Cycle covers of G(n) are counted by the permanent of its adjacency matrix A. Gurvits and Samorodnitsky give a lower bound that holds for any matrix and any doubly stochastic B: Let A = (a_ij) be a nonnegative matrix and let B = (b_ij) be a doubly stochastic matrix, then log per(A) is at least Σ b_ij log(a_ij/b_ij) + Σ (1−b_ij) log(1−b_ij). This builds on Gurvits's Bethe lower bound, derived from Schrijver's permanental inequality. There is no randomness anywhere: every doubly stochastic B supported on the edges of G(n) gives a valid lower bound.

I took B to be the actual Sinkhorn scaling of A_{G(n)}:

| n | (bound − (n/2) log n)/n | C − 1 |
|---|---|---|
| 1,000 | −1.60375 | −1.63254 |
| 4,000 | −1.61752 | |
| 16,000 | −1.62498 | |
| 64,000 | −1.62871 | |

The gap halves each time n quadruples, so it closes like n^{−1/2}, and the limit is the constant from the continuum Schrödinger problem. So for the real graph, log per(A) ≥ (n/2) log n + (C−1)n − O(√n) numerically. This also gives deterministic evidence that the residue structure doesn't shift the constant.

A proof of this step looks within reach. Take the ansatz p_ij ≈ a(i/n) a(j/n)/√n on the edges. Its row sums are sums of a smooth function over the squares in an interval, Σ_k a((k² − i)/n), so Euler–Maclaurin controls them and no residue classes appear. What remains is a perturbation lemma that corrects the O(n^{−1/2}) errors in the row and column sums to make the matrix exactly doubly stochastic, without losing more than o(n) in entropy. I'd make this the first theorem to aim for:

  log(number of cycle covers of G(n)) = log n! − (n/2) log n + Cn + o(n), with the matching upper bound having constant B in place of C.

**Step 2 (few cycles).** Krivelevich bounds the cycle covers that have many cycles by counting short cycles in the graph and applying Bregman to what is left. For G(n), a cycle of length L is a solution of a system of alternating square equations (x₁+x₂ = a₁², x₂+x₃ = a₂², …) with ordering constraints. Counting these is classical Diophantine work, and it involves residues only mildly. A spectral version should also work: in a random cover, the expected number of cycles is roughly Σ_L tr(Pᴸ)/L. The residue characters (mod 4 and so on) contribute only finitely many eigenvalues of constant size, so plausibly the number of cycles is O(√n), comfortably o(n/log n).

**Step 3 (merging cycles by rotation) is where the arithmetic really matters.** G(n) has independent sets of linear size, so the proof needs a residue-aware expansion lemma, as on the talk slide. This is the genuinely new number-theoretic ingredient.

**The upper bound is the weaker side.** Bregman is deterministic but gives only B. The general comparisons between the permanent and its Bethe approximation lose an exponential factor: Bethe(A) ≤ per(A) ≤ 2^{n/2} Bethe(A), and Anari [3] and Narang and Perkins [18] improved the factor to c^n for some absolute constant c < √2. We need e^{o(n)}. The girth-dependent improvements don't help either, because G(n) has triangles, e.g. 6 + 19 = 25, 19 + 30 = 49, 30 + 6 = 36. A sharp upper bound would need a "large-degree Bethe-exactness" theorem for deterministic matrices, which I don't know to exist. So a realistic target is a lower bound with constant C and an upper bound with constant B, leaving a gap of 0.0175 per vertex.

Would you like me to add this as a section of the paper,
"Deterministic route for G(n)"? It would include the
Gurvits–Samorodnitsky bound, the table above, and statements of the
three lemmas, together with a `bethe.py` script in
`hamiltonian/bounds/`.

Question
========
Yes, add this section to the paper. Also, since I (and a number of the
readers) may not be familiar with the details of the graphop and
sinkhorn argument, I'd like to add an Appendix which fleshes it out
(perhaps 5 to 6 pages).

Answer
======
I'll document the code addition first, then write the module, the section and the appendix.

The row-sum error of the continuum ansatz shrinks like n^{−1/2}, which is what the section needs. Now the new section and the appendix:

Now the appendix:

Checking two numbers I put in the appendix before compiling:

I've added the new section on a deterministic route for G(n) (§7, pp. 8–11) and the appendix explaining graph limits, permutons and Sinkhorn (Appendix A, pp. 11–15). The paper is now 16 pages. Both are kept as separate files and read into the main file.

**§7, a deterministic route for G(n)**
- **Cycle covers (§7.1):** the Gurvits / Gurvits–Samorodnitsky Bethe bound is stated as a theorem. Its corollary: any doubly stochastic matrix supported on the edges of G(n) gives a lower bound for the log of the number of cycle covers. The section includes the table of values from the actual Sinkhorn scaling for n up to 64,000, which approach C−1 at rate about n^{−1/2}. The proved upper bound has constant B and comes from Bregman's theorem.
- **Why the discrete scaling tracks the continuum (§7.2):** a new Lemma 7.3 with a full proof. Sums of a function of bounded variation over the squares in an interval equal √n·∫ f κ plus O(1). No residue classes enter, which is why the continuum guess a(i/n)a(j/n)/√n is nearly doubly stochastic. The table also records the measured row-sum error of that guess.
- **Problems L1, L2, L3:** stated precisely as Problems 7.4–7.6.
  - L1 is a flow-correction problem.
  - L2 reduces to counting short cycles, i.e. Diophantine systems in squares.
  - L3 is the residue-aware expansion needed for rotations.
- **The upper bound (§7.5):** explains why it is the weaker side.

**Appendix A**
- **Graph limits:** dense graphons and cut distance, then the sparse L^p theory and exactly why G(n) breaks it.
- **The profinite integers:** Haar measure, and the measure of the squares in ℤ_p, which is p/(2(p+1)) for odd p and 1/6 for p = 2. So the squares are Haar-null.
- **A joint equidistribution proposition, with proof:** it counts edges by interval and residue class. From it the limit operator Tf(x,α) = ∫ κ(x,y) E_{s∼σ} f(y, s−α) dy follows, and graphops are introduced informally around it.
- **Permutons and their large deviations:** the LDP plus Varadhan's lemma, with a truncation that handles the corner singularity of κ. This makes the first-moment constant a theorem for the random model, Proposition A.4, and the appendix spells out which two steps of §5 remain heuristic.
- **The Schrödinger problem:** the Gibbs principle, existence and uniqueness, the product form a(x)a(y)κ, symmetry, the value C = −2∫ log a, and a dual min–max formula. Any trial function gives an upper bound on C and any permuton a lower bound.
- **Sinkhorn:** convergence via Birkhoff–Hopf contraction in Hilbert's projective metric, the truncation estimate C − C_M = O(M^{−2}), and the numerics.
- **The discrete side:** matrix scaling, van der Waerden versus Bethe, and the explicit asymptotic computation. It ends with "where the residues went": the arithmetic coupling already has uniform marginals, so it contributes 0 to C.
- **A dictionary table** matching each continuum object to its finite-n counterpart.

Ten new references are added, and the bibliography is re-sorted. Three of them come from my own knowledge rather than a check against the source: Csiszár (1975), Birkhoff (1957) and Sinkhorn–Knopp (1967).

**Code:** `hamiltonian/bounds/bethe.py` produces the §7 table and the row-sum errors. I described it in `doc/bounds_plan.md` before adding it.

Files are in ~/Programming/Combinatorics/hamiltonian:
- doc/square_sum_bounds.tex
- doc/square_sum_bounds.pdf
- doc/sec_deterministic.tex
- doc/app_limits.tex
- doc/bounds_plan.md
- hamiltonian/bounds/bethe.py

Question
========
Can the Hamiltonian Permanent, as described starting on page 7 of the
attached paper play a useful role?

Answer
======
Yes, I think it could be quite useful, though not as an algorithm. Its value is that it turns counting Hamiltonian cycles into extracting one coefficient of a single generating function. That generating function can be attacked with a saddle-point (circle-method) analysis whose saddle point is exactly our Sinkhorn scaling. That would bypass the weakest part of the proof route in §7, Problems L2 and L3.

**What it gives us.** Barvinok defines hamp(A) = Σ_{g∈Hₙ} ∏ a_{i,g(i)}, the sum over permutations that are single n-cycles. Lemma 4.2 (p. 8) proves the identity

  hamp(A) = (1/n) · [x₁⋯xₙ] Tr(D(x)A)ⁿ.

For a 0/1 matrix, hamp(A) is the number of directed Hamiltonian cycles. The same argument handles paths, which is what H(n) counts:

  2H(n) = [x₁⋯xₙ] 𝟙ᵀ(D(x)A)^{n−1}D(x)𝟙.

Compare the permanent: per A = [x₁⋯xₙ] ∏ᵢ (Ax)ᵢ, which counts all cycle covers. That difference is the point. The route in §7 goes through cycle covers and then has to show that covers with few cycles dominate (L2) and that cycles can be merged (L3), and L3 is where the arithmetic causes real trouble. The trace formula only ever sees single cycles, so neither step is needed.

**Computationally it doesn't help.**
- Algorithm 4.3 is O(n⁴2ⁿ) with polynomial space, so n = 75 is far out of reach. Your frontier DP is much better.
- The fixed-rank algorithm (4.5) doesn't apply: A_{G(n)} has essentially full rank.
- The random model's mean matrix pᵢⱼ = 1/(2√(i+j)) is a positive Hankel matrix that is numerically low-rank. Its Hamiltonian permanent is *exactly* the expected number of Hamiltonian cycles of the random model. But the rank needed grows with n, and the cost n^{O(r²)} rules this out.

**Theoretically it suggests a circle method for Hamiltonian cycles.** Write the coefficient as an integral over the n-torus:

  hamp(A) = (1/n) ∫_{𝕋ⁿ} Tr(D(r e^{iθ})A)ⁿ ∏ⱼ (rⱼ e^{iθⱼ})⁻¹ dθ/(2π)ⁿ.

Take rⱼ = dⱼ², where S = DAD is the Sinkhorn scaling. Then D(r)A is similar to S, which has spectral radius 1, so the integrand near θ = 0 is about (∏dⱼ²)⁻¹ · λ₁(θ)ⁿ e^{−iΣθⱼ}.
- **Major arc.** The integrand doesn't change along the diagonal θ → θ + c𝟙. On the complementary directions, a Gaussian approximation should give a factor n!/nⁿ · e^{o(n)}. The e^{o(n)} needs the log-determinant of the relevant quadratic form to be o(n), which holds when ‖S‖_F² = o(n), and for G(n) we have ‖S‖_F² = O(√n). The result is

  log hamp(A) ≈ −log n + log(n!/nⁿ) − 2Σ log dⱼ = log n! − (n/2) log n + Cn + o(n).

That is exactly the conjecture's main term, now for Hamiltonian cycles directly rather than for cycle covers. Sanity check on the complete graph, A = J − I: the formula gives about (n−1)!/e, while the truth is (n−1)!, so the error is an O(1) factor.
- **Minor arcs.** Away from the diagonal, cancellation has to make |Tr(D(e^{iθ})S)ⁿ| small. The danger points are θⱼ = 2πaj/q. Every closed walk v₁v₂⋯ with v_t + v_{t+1} = k_t² has Σv_t = ½Σk_t². So at those points the phases are Gauss-sum-like sums e^{πi(a/q)Σk_t²}. Their imperfect cancellation for small q is exactly the local factors ϱ_q. In other words, our "singular series" heuristic becomes a concrete minor-arc estimate.
- **The easy part is too weak.** Cauchy's bound alone (|coefficient| ≤ the maximum of the integrand on the torus) gives log hamp ≤ (n/2) log n + Cn. That is worse than Bregman, because it loses the factor eⁿ. All of the content is in the Gaussian and minor-arc analysis.

**Precedents and caveats.** Saddle-point analysis of torus integrals in n dimensions, with n → ∞ and a maximum-entropy saddle point, is the method of:
- Barvinok and Hartigan, counting contingency tables and graphs with a given degree sequence (Trans. AMS 2012; Random Structures Algorithms 2013);
- Isaev and McKay, cumulant expansions of such integrals (Random Structures Algorithms 2018).

These citations are from memory, so check them. Those results are mostly for dense matrices. Our degrees are about √n, which makes the Gaussian step plausible but the minor-arc bounds genuinely new. Controlling Tr(D(e^{iθ})S)ⁿ on the minor arcs is a spectral problem: λ₁(θ) has to drop by a margin ε for θ off the major arcs, uniformly in n. It is analytic number theory, but of a more standard kind than the residue-aware expansion needed for L3.

**A cheap test.** We have exact counts for n ≤ 74. The major-arc prediction with its explicit Gaussian correction, which uses the actual Sinkhorn scaling of G(n), predicts the O(1) residual in our data table directly. If the residuals match, that is good evidence the minor arcs contribute only e^{o(n)}. If they don't, the size of the mismatch measures the arithmetic correction.

I can work out the Gaussian factor, run that comparison, and if it
holds up add a short subsection to §7 presenting this as an
alternative to L2–L3.

Question
========
Work out the Gaussian factor and run the comparison. Add a subsection
if it holds up.

Answer
======
I'll start by deriving the prediction, then test it against your exact counts.

That first refinement is far off, so I'll try the natural alternative weighting before drawing conclusions:

Citation confirmed. Now the plan addendum and the module:

Confirmed: for n ≤ 30 the scaling degenerates, with some entries going to 0 and d to ∞, so A lacks total support. From n = 31 on it is well behaved. Now the subsection:

Checking two numbers I put in the text:

Both fixes are in. Committing to your repo:

The Gaussian factor did not hold up, but the zeroth-order version of Barvinok's approach did. A predictor built from the Sinkhorn scaling of each actual G(n) fits your exact counts much better than the constant-C formula. I added it as §7.6.

**An exact identity.** Rescale A to its doubly stochastic Sinkhorn form S = DAD. Then Barvinok's path identity becomes exactly

  2H(n) = n·(∏ d_j)⁻² · E[d_{X₁}d_{Xₙ}; X₁,…,Xₙ distinct],

where X is the stationary Markov chain with transition matrix S. Replacing the probability that the chain never revisits a vertex by its i.i.d. value n!/nⁿ, and decoupling the endpoints, gives the predictor

  Ĥ(n) = (n/2)·(n!/nⁿ)·(∏ d_j)⁻²·(mean d)².

Asymptotically this has the same constant C, but it uses the actual scaling of each G(n).

**What failed**
- **The Gaussian factor itself.** The visit counts are lattice variables with mean 1, so their local probabilities behave like Poisson (e⁻¹ each), not Gaussian ((2π)^{−1/2} each). The saddle-point approximation is therefore wrong at exponential order, by about 0.081n in the logarithm even for the complete graph.
- **First-order "loop" corrections.** I tried the pair-collision factor and the det′(I − S) factor. They shift log Ĥ by −23 to −29 and by −8 to −10, when the correction actually needed is about +2. At these sizes the degrees are only 2–7 and S has eigenvalues down to −0.95, so nothing here is a small perturbation.

**What held up**

| Residual of log H(n), 31 ≤ n ≤ 74 | Standard deviation | Range |
|---|---|---|
| Constant-C formula | 0.72 | [−2.7, 0.9] |
| Scaling predictor Ĥ(n) | 0.23 | [1.89, 3.11] |

The new residual is smooth: it moves by at most 0.6 between consecutive n, and usually by less than 0.15. The dips of the old residual near n = 40, 58–60 and 70 are explained entirely by the scaling vector of A_{G(n)}. What's left is a slowly varying offset of about 2, consistent with e^{o(n)}. For n ≤ 30 the scaling degenerates, because A_{G(n)} lacks total support there.

**The case against H(75) is now much stronger.** Against Ĥ its residual is +6.25, a jump of +4.3 from a curve that otherwise never moves more than 0.6 per step.

**What the subsection adds as an open problem.** Everything beyond Problem L1 now reduces to one statement, Problem L*: for this Markov chain, the log of the probability that X₁,…,Xₙ are all distinct is log(n!/nⁿ) + o(n). Together with L1, that would prove the conjecture for paths directly, without going through cycle covers. The data suggest the ratio is in fact e^{O(1)} in this range.

**Code and reference.** The new script is `hamiltonian/bounds/hamp_predictor.py`; it also writes the figure and evaluates the two failed corrections. Before adding it I described it in `doc/bounds_plan.md`. I added Barvinok, Math. Oper. Res. 21 (1996), 65–84 to the bibliography, checked against the publisher's record.

Files are in ~/Programming/Combinatorics/hamiltonian:
- doc/square_sum_bounds.tex
- doc/square_sum_bounds.pdf
- doc/sec_deterministic.tex
- doc/figs/predictor.pdf
- doc/bounds_plan.md
- hamiltonian/bounds/hamp_predictor.py

Question
========
The writeup of the paper refers to various files and programs. That
may be fine for someone whose looking at the whole repo. However, if
the paper is to be submitted for publication, that doesn't work.  I
know that some publications allow one to attach other relevant files,
but that always doesn't hold. Can you write a separate, necessarily
longer, version in which you discuss the data and algorithms (which
may be written in informative pseudocode) that you refer to in the
current paper?

Answer
======
I'll start by finding every place the current paper refers to files or programs.

Few references, all easy to replace. Most of the work is the new material. First I'll regenerate every data set the long version will tabulate, starting the slow random seed search in the background:

Now the glue arrangements I'll tabulate, the exhaustive t = 3 check, and the exhaustive seed statistics:

The random seed search has finished: k = 2 at n₀ = 45, 55, 63, 65 and 69, and 45 is still the best ratio. I have all the data now. First, the plan addendum:

Now the generator for Appendix C. It recomputes or checks every number it prints:

While that runs, I'll write Appendix B:

Two fixes: the position convention in the certificate algorithm, and a softer statement about the memory used for n = 75.

Appendix C generated with all its checks passing. Now I'll assemble the long version from the short one:

The hang was a LaTeX trap in the generated glue tables: a line break `\\` followed directly by `[` is read as an optional spacing argument. I'll fix the generator and patch the generated file:

A corrupted bookmark file left by the killed run; removing it:

It compiles: 26 pages. Two cleanups: one overfull line, and one character that forced a bitmap font (bad for a submission). I'll find both:

Two Type 3 font sources remain: the matplotlib figure, which I'll re-export with embedded TrueType fonts, and one bitmap font. Locating the latter:

Layout is reasonable. I'll inspect two pages closely:

Moves that span two lines lose their alignment. I'll fix the generator and regenerate Appendix C:

Good. Committing everything:

I've written the separate long version, `square_sum_bounds_long.tex` (26 pages). It doesn't mention a single file, program or module. The short version is unchanged apart from two small fixes noted below.

**What changed from the short version**
- Every reference to a file or program now points to an algorithm or a data table instead.
- The "Software" section is now "Computations". It is a table mapping each computational claim to the algorithm that produced it and the data that supports it.
- The abstract says the paper is self-contained.

**Appendix B: Algorithms.** Eleven procedures in pseudocode, divided into two kinds.
- **Exact, integer-only:**
  - the frontier dynamic program that counts paths;
  - the lift;
  - the glue search;
  - the search for moves, with a short proof that the greedy selection is optimal;
  - certificate verification, with the index bookkeeping for reversed segments;
  - the seed search;
  - the Bregman bound and the edge count, using exact degree and edge formulas.
- **Numerical, double precision:** the continuum Sinkhorn computation of C, the Sinkhorn scaling of G(n) with its Bethe value, and the scaling predictor. Their convergence residuals are reported, and the appendix states that they support evidence only, never a theorem.

**Appendix C: Data**
- The full table of H(n) for 15 ≤ n ≤ 75.
- Complete certificates: the three seed paths with each move's window, anchors, and old and new contents; a table of the lift checks for two levels; and four more glue arrangements (for the other endpoint pairs and t = 5, 7), each with all junction sums checked.
- Seed-search tables, with exhaustive values of k(n) for odd n ≤ 41.
- The numerical tables for C, edge counts, the Bregman and Bethe bounds, and both residuals for every n.

Every number in Appendix C was generated and checked by `hamiltonian/bounds/long_data.py`, not transcribed by hand. I described that script in `doc/bounds_plan.md` before writing it.

**Please check one passage.** The description of the counting DP (Algorithm 1, start of Appendix B) is a generic frontier/"mate" dynamic program in the style of Knuth's SIMPATH. Please make it match your actual implementation: the ordering heuristic, the state encoding, and how states are merged in parallel. Also check the sentence in C.1 about where the counts came from (and whether any were cross-checked against OEIS). Both spots carry a `%% CHECK` comment in the source. I gave the Knuth and Kawahara et al. references from my own knowledge, without checking them against the sources.

**Fixes made along the way**
- Fonts that journals reject:
  - Both versions: "Pósa" is now written as `P\'osa` instead of a raw accented character.
  - The predictor figure now embeds TrueType fonts instead of Type 3, which many journals reject.
  - Long version only: it now uses T1 fonts with Latin Modern, so the PDF has no bitmap fonts.
- The data generator now avoids a LaTeX trap: a line break followed by `[` hung the compilation.

Files are in ~/Programming/Combinatorics/hamiltonian:
- doc/square_sum_bounds_long.tex
- doc/square_sum_bounds_long.pdf
- doc/app_algorithms.tex
- doc/app_data.tex
- doc/app_limits_long.tex
- doc/square_sum_bounds.tex
- doc/square_sum_bounds.pdf
- doc/figs/predictor.pdf
- doc/bounds_plan.md
- hamiltonian/bounds/long_data.py
- hamiltonian/bounds/hamp_predictor.py
- data/bounds/seed_search_exhaustive.txt
- data/bounds/seed_search_random.txt
- data/bounds/glue_other.json

Question
========
For the exact calculations I didn't use SIMPATH. At first I used the
package graphillion which implements a version of the SIMPATH
algorithm. On advice from the author, Minato, I determined a good
vertex order by implementating the exact pathwidth algorithm of
Coudert. I found that this was running out of storage at about n=56. I
then changed to using a dynamic programming algorithm based on
Held-Karp, but with using the exact pathwidth order. I found later
that this was laid out in
@article{pettersson2014enumerating,
  title={Enumerating hamiltonian cycles},
  author={Pettersson, Ville H},
  journal={The Electronic Journal of Combinatorics},
  pages={P4--7},
  year={2014}
}
I had to do a lot of work (with Claude) to optimize use of the storage
hierarchy, and use parallel processing.

Answer
======
Your repo has `doc/hampath_paper_full.tex`, which presumably documents the counting computation itself. I'll describe the algorithm from it rather than from my guess:

Now the bibliography and the provenance sentence in Appendix C:

The layout is fine apart from two comments that wrap awkwardly. Shortening them, then committing:

I've rewritten §B.1 and Algorithm 1 of the long version to describe the method you actually used. I based them on your engineering write-up, `doc/hampath_paper_full.tex`, rather than on my earlier guess.

**§B.1 now describes two stages**
- **Graphillion stage (n ≤ 56):** frontier-based zero-suppressed decision diagrams (the SIMPATH approach), using an optimal vertex-separation ordering as Minato advised. That ordering came from Coudert's integer-programming formulation solved exactly as a weighted MaxSAT. The text says this ran out of storage at about n = 56.
- **Dynamic programming stage:** a Held–Karp-type recursion restricted to a path decomposition along the same ordering, i.e. Pettersson's approach, which the text says you found afterwards. It gives the widths w = 11 at n = 57 and w = 13 at n = 59.

**Algorithm 1 is now vertex-by-vertex**, matching your implementation.
- The state encoding follows your write-up: each frontier vertex has status 0, −1 (interior) or a chain label, and there is a closed-chain counter c ∈ {0,1}. Labels are canonicalised, so there are at most Bell(w) patterns.
- The text explains why c is needed: without it, phantom states arise, and n = 15 gives 11 instead of 1.
- The edge sweep for each vertex covers all subsets X of its earlier neighbours at once and is fused with the eliminations that follow, as in your v4.
- An "ordering and engineering" paragraph covers the simulated-annealing refinement of the ordering, the packed 64-bit hash-table keys, and the 16 GB machine being enough for n ≤ 59.

**Please check this sentence.** I wrote that the runs up to n = 75 used a parallel version that partitions the state tables among worker processes, uses the storage hierarchy deliberately, and merges with deduplication at each step, on the 1 TB / 64-core machine. That is reconstructed from fragments of our earlier conversations, not from a document, and it carries a `%% CHECK` comment in the source. The paragraph also cites your engineering paper as "in preparation".

**Other changes**
- The C.1 sentence now says the values for n ≤ 56 were first obtained with Graphillion and that the dynamic program extended them to n = 75.
- The short version now has one sentence on how the counts were computed, citing Pettersson and Coudert.
- I removed the Kawahara et al. reference, which no longer applies, and added these:
  - Pettersson, Electron. J. Combin. 21(4) (2014), #P4.7. You gave the number as "P4--7"; I believe it is P4.7, so please confirm.
  - Coudert's report, Graphillion, Minato (1993), Kinnersley (1992), Bellman (1962) and Held–Karp (1962), with the details taken from your engineering paper's bibliography.
  - Your engineering paper.

Files are in ~/Programming/Combinatorics/hamiltonian:
- doc/square_sum_bounds_long.tex
- doc/square_sum_bounds_long.pdf
- doc/app_algorithms.tex
- doc/app_data.tex
- doc/square_sum_bounds.tex
- doc/square_sum_bounds.pdf
- hamiltonian/bounds/long_data.py

Question
========
I've also reorganized: I put all the tex and pdf files in doc/Papers
. The engineering part also has an overview in doc/Talks/talk2.tex
. Also, I like to have all the references in a separated bibtex file.

Answer
======
I'll look at the new layout first.

Identical to my copies, so I'll work locally in the same layout. Extracting all entries from the long version, which contains every reference the short one uses:

Now the BibTeX file. It has the 54 existing entries plus one new one: Fagin et al. (1979) on extendible hashing, cited in the corrected engineering paragraph.

Now switching both papers to `\bibliography{square_sum}`, fixing the figure paths, and correcting the engineering paragraph from your talk:

Both compile cleanly. The two BibTeX warnings are just entries with no page numbers: an article-number journal and a paper I couldn't confirm pages for. Let me check how the reference list renders:

The references render correctly. Before committing I'll recheck that you haven't edited the files on your machine in the meantime:

Committed. Last, a check that both papers build on your machine in their new location:

Your two requests are done. Both papers now use a separate BibTeX file, `doc/Papers/square_sum.bib`, and build in their new location. I also corrected the description of the counting computation using your engineering talk.

**Separate bibliography.** `square_sum.bib` holds all 55 references and is shared by both versions, which now end with `\bibliographystyle{amsplain}` and `\bibliography{square_sum}`. Build each one as pdflatex, bibtex, pdflatex, pdflatex.
- Fifty-four entries are the existing ones. The new one is Fagin, Nievergelt, Pippenger and Strong (1979) on extendible hashing, cited in the corrected engineering paragraph.
- BibTeX gives two harmless warnings: Borgs et al. (2018, J. Mach. Learn. Res.) has an article number rather than pages, and I couldn't confirm the page numbers for Starr (2009).
- amsplain doesn't print the doi fields; they are still in the file if you switch to a style that does.

**Paths after your move**
- `\input{figs/gluing.tex}` in both papers and `figs/predictor.pdf` in `sec_deterministic.tex` now point to `../figs/`.
- The data generator `long_data.py` now writes `doc/Papers/app_data.tex`.
- I recorded the new layout in `doc/bounds_plan.md`.

**Engineering paragraph, rewritten from `talk2.tex`.** It now says:
- 64-bit state keys with 128-bit counts.
- At each step the work is merging a stream of (key, count) pairs into the next table, which is an extendible hash table, so it grows with the actual output.
- New pairs go first into a radix-sort buffer keyed by a hash prefix, so each bucket is loaded into cache once per flush rather than once per insertion.
- Buffer and bucket sizes are chosen from the machine's cache sizes, and freed slabs are returned to the OS explicitly.
- The headline example: n = 61, with width 14 and a peak of 620 million states, ran in 97 minutes in under 2 GB on an M3 Pro.

Your talk says nothing about the parallel version, so the paragraph only says the cases up to n = 75 were run with a parallel version on the 1 TB / 64-core machine. A `%% CHECK` comment marks where you could add one sentence on how that version splits the work.

**Build check.** I confirmed both papers build here, with no errors and no undefined references: 18 pages for the short version, 27 for the long one. On your machine the short version also builds cleanly. The long version stopped there only because the sandboxed Linux shell I use on your computer lacks the `algorithms`/`algorithmicx` packages. Those are standard in a full MacTeX install, so it should build for you, but I couldn't confirm that.

Your Papers and Talks folders now hold the papers and `talk1`/`talk2`, but my beamer talk is still in `doc/talk/`. Its paths, like `../figs/gluing.tex`, still work there if you want to move it.

Files are in ~/Programming/Combinatorics/hamiltonian:
- doc/Papers/square_sum.bib
- doc/Papers/square_sum_bounds.tex
- doc/Papers/square_sum_bounds.pdf
- doc/Papers/square_sum_bounds_long.tex
- doc/Papers/square_sum_bounds_long.pdf
- doc/Papers/app_algorithms.tex
- doc/Papers/sec_deterministic.tex
- doc/bounds_plan.md
- hamiltonian/bounds/long_data.py

Question
========
I should update talk2. It was written before I pushed the calculations
through n=75 (even though that calculation should be checked).  I see
that hampath_paper_full also isn't updated to reflect the
parallelism. You can see that implemented in the code in
hamiltonian/claude . The main idea is that the contributions to the
updated sum can be done in parallel. There was some effort to balance
the work as it isn't obvious from just looking at the subset how to
balance it. Can you look at the the various markdown files, and the
code in hamiltonian/claude to extract the algorithms?

Answer
======
I'll start by surveying the code and the notes.

The current code is `hamiltonian/claude/ham_dp_c.c` (138 KB, dated April 22). The session notes are inside `files.zip` archives in `doc/Claude/*`. I'll list which markdown files those contain:

The zip archives hold only code snapshots. The notes that mention parallelism are `doc/new_claude.md` and `doc/lower_bound.md`:

The `count_ham_paths_sm` backend (from line 1266 on) is the parallel code. I'll read it section by section, starting with its header comment and the parallel merge:

Note that runs are sorted by hash value, not by raw key, which is what makes the hash-range splitting balanced. Next, the work-stealing queue, run flushing, and the worker loop:

Shards here are static, `[i·chunk, (i+1)·chunk)`. Yet a work-stealing queue and `SM_STEAL_FACTOR` are defined, so I'll find where they're used:

The queue is defined but never called; the final code shards inputs statically. The session log should say what happened to it:

The work-stealing queue doesn't appear in the log's account of the final design. I'll check the older code snapshots for where it was actually used:

Before the C was split into its own file it was embedded in `ham_dp_c.py`, so the stealing may have been used in those earlier versions:

The queue is defined in every snapshot but never called. One final check across all the snapshots:

Overflow is detected, and the Python wrapper raises an error rather than returning a wrong number. I'll check for any unflagged u64 additions, and look at what the n = 75 log recorded:

There are 14 profile logs for n = 75. If several of them finished, comparing their results is a direct check on the suspect value:

I wrote up the parallel algorithm in `doc/parallel_dp_algorithms.md`. Your summary of the main idea is right. A source state together with a subset of the new vertex's back-edges gives at most one contribution to the next table. The only thing that couples different sources is adding up the counts for equal keys. So the code generates contributions in parallel, sorts them, and merges with dedup in parallel. No shared table is written during the parallel phases, and no counts are updated with atomics.

**The balancing.** It's simpler than it looks from the code. Every run, and the merged table itself, is sorted by `eh_hash(key)`, which scrambles the keys. So a contiguous slice of the table is effectively a random sample of states, and the static slices `[i·chunk, (i+1)·chunk)` get about the same mix of heavy and light states. The merge splits hash space using sampled splitters. Equal keys have equal hashes, so dedup stays exact inside each range.

**What the code and logs show:**
- **Balance at n = 75 (63 workers).** The slowest worker took 1.03–1.19× the mean time on the heavy steps, and up to 1.9× on the light steps, which cost little in total.
- **A balancing bug found along the way.** An earlier version sorted runs by raw key but split the merge in hash space, which left 5 of 6 merge threads idle. The fix was to sort everywhere by `eh_hash`.
- **The work-stealing queue is never used.** `SMWorkQueue`/`sm_wq_claim` appears in every snapshot but is never called, so the `steal_factor` setting has no effect.

**H(75) needs a recheck.** There is only one completed n = 75 run. It is also the first run where two untested limits are reached:
- The frontier width is 15, the most the 64-bit key can hold.
- The total exceeds 2⁶⁴. Per-state counts are u64 with an overflow flag, and totals are u128.

The document proposes cheap tests for both on small n with known counts. Without a big machine, those tests can check the code paths but can't independently confirm the H(75) value.

There are also two small discrepancies to settle before the papers and talk are updated:
- Your note says the machine had 1 TB, but `config/n75_linux.yaml` says 768 GB.
- Code comments say entries are 24 bytes; they have been 16 bytes since the count was changed from u128 to u64.

If the extraction looks right to you, the next step is to update
`talk2.tex` and `hampath_paper_full.tex` from it, and to fill the `%%
CHECK` sentence in `app_algorithms.tex`.

Question
========
The 768GB is correct. As you can see from the files in data/linux only
`profile_76log.txt` was a finished run. Previous attempts all aborted
because of running out of storage. I had to hand-tune the machine.yaml
file until I could get the run to finish. As you can see from the log
file, I used the sort/merge variant because that has the best
locality.  Everything you've noted looks correct.

Answer
======
Now I'll update the markdown note to match, then rebuild the long paper.

I've made the corrections in the long paper and in `doc/parallel_dp_algorithms.md`. There's no `profile_76log.txt` in `data/linux`, so I took you to mean `profile_75log.txt`, the one that ends with `ham_paths=652211775281872067084`.

- **Long paper, Appendix B.** The machine is now 768 GB and 64 cores. The `%% CHECK` comment is replaced by a short paragraph on the parallel sort/merge version. It covers how work is split across workers, why slices of a hash-ordered table balance the work, and the hash-range merge. It also says plainly that the n = 75 run is the only one to reach frontier width 15 and the only one whose total exceeds 2⁶⁴, and that it hasn't been repeated. The PDF is rebuilt at 27 pages.
- **The note.** It now says the n = 75 run used a hand-tuned `machine.yaml`. It also records what the other 13 n = 75 logs show. They are aborted attempts, and several reached steps with width 15 before running out of storage. Since none produced a total, they don't test whether the width-15 encoding is correct. Every completed run for n ≤ 74 has maximum width 14, so the cheap width-15 test on a small n is still the most useful check.

Next is updating `talk2.tex` and `hampath_paper_full.tex` along the
lines of section 7 of the note.

Question
========
Do the next step of updating `talk2.tex` and `hampath_paper_full.tex`

Answer
======
I'll start by reading both files' structure.

Now the new talk frames.

Talk is done. Now the paper sections.

I've updated both the talk and the paper, built them, and saved them back to your repo along with the bibliography.

The value for n = 75 looks wrong, not just untested. H(75)/H(74) ≈ 300, while H(n)/H(n−1) stays between 1.3 and 6.4 for every n from 62 to 74. That also matches the scaling predictor putting H(75) about 70 times lower. Both documents now call H(75) provisional, and give this jump as a reason along with it being the first run at frontier width 15 and the first total above 2⁶⁴. It's the first thing I would check.

**`talk2.tex`** (now 31 slides, up from 22):
- **New Part VI, "Going Parallel"** (6 slides):
  - why the two hash-table attempts failed;
  - a diagram of how workers generate contributions, sort them into runs and merge them;
  - workers, runs and the decision to keep runs in RAM or spill them to SSD, with the memory bound;
  - the parallel merge by hash range;
  - load balancing by storing tables in hash order, with the measured worker-time ratios and the raw-key bug;
  - correctness hazards (the `MADV_FREE` silent data loss, the 64-bit count guard) and which cross-checks passed.
- **Results and lessons:**
  - a table of H(61)–H(75) with run times on the 768 GB machine;
  - a "Lessons from Going Parallel" slide, which mentions your hand-tuning of `machine.yaml` and that only one of the 14 attempts at n = 75 finished.
- **Outline, subtitle and closing slide** now cover the parallel version.

**`hampath_paper_full.tex`** (now 18 pages):
- **Abstract, introduction and conclusion** now run to n = 74, with n = 75 provisional.
- **New section on extendible hashing** (the version that reached n = 61 on the laptop), summarising the material from the talk.
- **New section on the parallel sort–merge version:** earlier attempts, why contributions are independent, one step in detail, the merge by hash range, load balancing, safeguards and tuning.
- **Counts table** now covers n = 56–75, with a paragraph on n = 75 and the cheap checks that need no large machine.
- **Bibliography** is now in `square_sum.bib`. I added entries for Giscard–Rochet and htd. Four references the text never cites (Minato, Kinnersley, Bellman, Held–Karp) are kept in the list with `\nocite` so it doesn't shrink.

The paper still has Type 3 fonts and three overfull lines. Both were
there before my edits; switching to `lmodern`/T1 would fix the fonts.

Question
========
Fix the fonts

Answer
======
Your copy on disk is unchanged since my last write, so it is safe to overwrite.

The paper's fonts are fixed: I added `\usepackage[T1]{fontenc}` and `\usepackage{lmodern}`, and the PDF now embeds only Latin Modern Type 1 fonts. `talk2.pdf` had no Type 3 fonts, so I left it alone.

I also fixed the three overfull lines that were already there:
- I added break points in `apply_elim_seq` in the fused-sweep list.
- I narrowed the gap in the cost-function equation from `\qquad` to `\quad`.
- I set the cost-function history table in `\small`.

The paper still builds cleanly at 18 pages, and both the `.tex` and
the PDF are saved in `doc/Papers`.

Question
========
One peculiarity that I don't quite understand is why the run for n=74
took much less time than the runs for n=62 through n=73. There was
another user on the machine during all that time, so it might have
just been because we were fighting for resources.

Answer
======
Let me look at the logs before guessing.

All four n=74 runs did exactly the same work, which is informative. Let me compare their configurations.

The fast runs all date from after April 16, with the same 64 workers. Let me see what changed between April 13 and 16.

The "mlp" runs used the same ordering as the slow one (identical profile and identical work), so the ordering isn't the explanation. Let me compare a heavy step between the slow and fast n=74 runs.

The slow run's time went mostly into the introduce step, not into computing states. Let me confirm when the code changed, using the snapshots.

I've found the cause, and it also shows I got something wrong in the documents. Let me fix those before answering.

The n=75 run also used 64 workers (w0–w63), not 63 as I had written. I'll correct that too.

Competition from the other user doesn't explain it; the code changed. The runs for n = 62–73, and your first n = 74 run (3449 s), used the code of April 11–13. The three fast n = 74 runs (945, 957 and 1205 s) used the code from April 16 on. All four n = 74 runs did exactly the same work, about 1.31·10¹¹ state × subset combinations, so the comparison is like for like. Here is where the 2500 s went:

| | old run (Apr 13) | new run (Apr 16) |
|---|---|---|
| introduce | 1417 s | 12 s |
| merge | 1002 s | 240 s |
| flush | 581 s | 306 s |
| generating states | 345 s | 337 s |
| steps spilled to SSD | 3 | 0 |

- **Introduce.** The old code re-sorted the entire table by hash after every introduce, which took 40–50% of the run time for every n from 62 to 74. The sort was never needed: introducing a vertex never makes two keys equal, so nothing gets combined, and the workers don't need sorted input. Doing the introduce in place removed it.
- **Merge and flush.** Entries shrank from 24 to 16 bytes, so there is a third less data to move. As a result, the three heaviest steps (35, 49, 55) no longer spill to SSD. Step 35 alone went from 433 s to 56 s.
- **Generating states.** This is unchanged. It's the CPU-bound part, so it is where competition for the cores would show up most. In the April 17 run it is actually higher (496 s) and less evenly balanced across workers, which may be the other user.

Per unit of work, all the old runs take 21–28 ns and the new ones 7–9 ns. So n = 62–73 would presumably run about 3× faster with the current code.

This also showed that I had two things wrong in the documents, which I've now corrected and rebuilt:
- **The four n = 74 runs all used the same ordering.** I had written that they used different orderings. They check the code changes, not the ordering, so no n ≤ 74 has been checked across orderings either.
- **The n = 75 run used 64 workers, not 63.**

I also rewrote how the documents describe the in-place introduce. The
table ends up ordered by the hash of each previous key, not the new
one, which is still effectively random, so the load-balancing argument
holds. The talk's results table and the paper's counts table now show
3449 s for n = 74, so its time is comparable with n = 62–73, with a
note giving the 945 s from the later code. The paper also gives the
breakdown above.

Question
========
In `talk2.tex` it still mentions n=61 as a new record. This should
certainly be updated to at least n=74 (with an appropriate caveat
about the n=75 value).

Answer
======
The closing slide's wrapping is ugly; I'll fix it.

The talk now presents n = 74 as the record and calls n = 75 provisional:

- **The n = 61 slide** no longer says "New result … first time computed". The block is now headed "Laptop milestone": H(61) is the largest n done on the laptop, and the parallel sort–merge later took the exact counts to n = 74.
- **The results slide** is now titled "New Results: Exact to n = 74; n = 75 Provisional". It keeps the H(75) footnote and the "n = 75 needs a second run" box, which gives the ratio of about 300 and the one finished attempt out of 14.
- **The outline** now reads "Exact to n = 74" and "n = 75: provisional".
- **The closing slide** reads "n = 61 in 97 minutes on a laptop / exact counts to n = 74 on 64 cores / n = 75: one run, not yet confirmed".

The engineering paper already treated n = 74 as the extent and n = 75
as provisional, so it needed no change. The talk is still 31 slides;
the `.tex` and PDF are saved in `doc/Talks`.

Question
========
Look in the repo in ~/Programming/Combinatorics/kobayashi-pw . This
gives a new implementation of vertex separation/pathwidth that is much
more efficient than the MaxSat algorithm. It also generates a DAG
which encodes all of the possible minimal pathwidth orderings. It is
possible to run the exact optimization of the secondary cost function
over that DAG used by simulated annealing (see some of the markdown
files in its doc subdirectory). It would be useful if I could use the
functions from that repo in place of the MaxSat simulated annealing
precomputation. Can you work on that?

Answer
======
The DAG optimiser already exists in `orders.py`. Now the hamiltonian side: how orderings are produced and consumed.

SA actually uses `_sa_cost_fn` when one is loaded. Let me look at that model, since it's the one the n = 75 run optimised.

The pieces exist on both sides. Before writing the plan, I'll check the API and measure how the DAG route scales up to n = 75.

The connected shell is a small Linux VM and the `kpw` binaries are macOS builds, so I'll do the measurements in my own workspace instead.

DAG construction takes seconds. Now n = 71–75, where n = 75 has width 15:

n = 75 takes 3 s, against 1284 s for MaxSAT. Next, the Python V7 optimiser on these DAGs, checked against `_dp_cost` from your repo:

The V7 optimum matches your `_dp_cost` exactly. Now the larger n, in the background:

While that runs: what SA actually optimises when `sa_cost_params.json` is present.

All 19 orderings are recovered and reproduce the logged frontier sizes. Checking the V7 runs:

A notable result already. I'll get the actual work for each logged run, which is ground truth to test the cost models against.

For n = 57–60 the actual work is small enough to run the real DP here on both orderings. Staging the DP code:

Both orderings give the correct H(57). I'll queue n = 58–60 in the background, after the V7 optimisations finish.

The DP comparisons for n = 58–60 are still running. In the meantime I'll draft the design document.

Now the plan document, while n = 59–60 finish.

The n = 59 DAG run stopped at step 45 without an error message, which looks like the 7 GB container running out of memory. I'll see whether the SA run survives.

The prediction is badly miscalibrated (4.5·10⁶ predicted), but the ordering is genuinely good: 2.52·10⁸ actual work, 45% less than the SA ordering. Checking the count, then n = 58:

The n = 59 optimisation is slower: the work model produces Pareto fronts of hundreds of labels, unlike V7's single label. While it runs, I'll revise the plan around these results.

The n = 60 run was killed for running out of memory at step 42, after 3.4·10⁹ of work. I'll rerun with smaller worker buffers.

While n = 60 reruns, the revised plan:

Saving the plan and the experiment scripts to the repo:

I haven't changed any code yet. Following your preference, I've written a plan, `doc/kpw_ordering_plan.md`, backed by measurements and real DP runs. The DAG works well as a replacement for the pathwidth computation, but the measurements change which objective I'd optimise over it.

**Pathwidth and the DAG take seconds.** `kpw2 -c` builds the DAG of optimal orderings in 0.2–4 s for n = 57–75; at n = 75 it takes 3 s, against 2179 s for MaxSAT plus SA. Its pathwidth agrees with MaxSAT for every n from 57 to 75. The exact V7 optimum over the DAG matches `_dp_cost` to all printed digits. Loading the DAG and optimising it in Python adds 10–95 s.

**V7 is a poor objective.** I recovered the orderings actually used from the step logs; all 19 reproduce the logged frontiers. Then I ran the real DP on them and on the alternatives:

| n | SA ordering (logged) | V7 optimum on DAG | fitted-model optimum on DAG |
|---|---|---|---|
| 57 | 1 | 0.95 | **0.55** |
| 58 | 1 | 0.83 | **0.44** |
| 59 | 1 | > 1.19 | **0.65** |
| 60 | 1 | 1.14 | ran out of memory at step 43 |

Entries are total DP work relative to the SA ordering; lower is better. All completed runs give the correct H(n). The V7 run at n = 59 also ran out of memory, at step 45, after already doing 1.19× the SA ordering's total work.

- **Fitted model.** This is a least-squares fit of log(states_out/states_in) on 11 features of each step, from your 925 logged steps. It explains 88% of the per-step variance with each n held out. Predicted work fits the (a, e, k) form in `orders.py`, so it can be minimised exactly over the DAG. For each n I fitted it without that n's own run. Its optimum also roughly halves peak states at n = 58–59.
- **Its absolute predictions are off by 1–2 orders of magnitude.** It is only good for ranking orderings, and it knows nothing about memory: at n = 60 its ordering does less work through step 42, but then needed more memory than this 7 GB container has.
- **The MLP that SA currently optimises is unreliable.** It gave the ordering used for n = 75 the lowest predicted cost of all 19 runs, while that was by far the most expensive one. Its spill penalty never fires, because its predicted state counts are about 40× below the real peaks.

Everything above was tested only for n ≤ 60, on four graphs.

**The plan:**
- a new `ham_kpw.py` that builds and caches the DAG and returns exact optima;
- a new `work_model.py` with a script that fits the model from `profile_*.txt` logs;
- a `--ordering {maxsat,kpw,bfs}` option in `ham_count`, with options to polish the result with the existing SA or to rerank candidates by partial DP;
- tests;
- kobayashi-pw as an editable path dependency.

At n = 75 the DAG offers about 5.6·10³⁸ other optimal orderings, which makes the H(75) recheck easier to set up.

The experiment scripts and orderings are in `doc/kpw_plan/`. They hard-code paths on my machine, so they record what I ran rather than being ready to rerun.

Decisions for you:
1. Should the default objective be the fitted work model (my recommendation) or V7?
2. Should `--ordering` default to `kpw` now, or stay `maxsat` until you've compared at n ≥ 61 on a large machine?
3. Do you want a C optimiser added to kobayashi-pw? Exact optimisation under the fitted model takes 4–6 minutes per n in Python at n = 59–60, and likely longer at n = 75; in C it would be seconds.
4. Should the MLP annealing path stay behind `--ordering maxsat`, or
   be retired?
   
   
Question
========
All 4 are good. As far as 2 and 4, keep the old orderings available by
option for now. As far as the fitted work model that sounds good,
except that I notice that n=60 ran out of memory. Is there a, possibly
suboptimal, variant which uses less memory. Where is the memory use?
Is it another dynamic program? Orginally I thought that the DAG might
be like a BDD, in that the amount of memory necessary for optimization
was essentially the size of the DAG. Is this different?

Answer
======
Before answering, let me check two things: what step 43 of the n = 60 ordering looked like, and how much memory the DAG optimisation itself uses.

The counts at step 43 explain the failure. Now the optimiser's own memory, measured at n = 58 for both objectives:

The memory that ran out at n = 60 was the Hamiltonian-path counting DP (`ham_dp_c`), not the ordering optimisation. This container has only 7 GB.

**What the counting DP ran out on.** With the fitted ordering, step 43 starts from 64M states and adds a vertex with at least 3 back-edges. That is up to about 5·10⁸ raw contributions before deduplication, around 8 GB, so the step spills to disk and the run was killed. The V7 ordering got through a step almost as large at step 42 (62.6M states, 3 back-edges), but that step's output was only 27M states thanks to eliminations. So a step's memory depends on both its raw contributions and the size of its output, and the work objective penalises neither.

**The optimisation over the DAG is another dynamic program, and it is BDD-like up to the number of labels per node.** It is linear in the DAG, but each node stores a set of labels rather than one value:

- **V7 (and V2):** one label per node, so memory is essentially the DAG's size. At n = 58 the run used 95 MB after loading the DAG (45k nodes, 321k arcs) and 110 MB after optimising.
- **The fitted work model:** a label is the pair (cost so far, predicted state count). If one partial ordering is cheaper so far and another has fewer predicted states, both must be kept, because either can win later. That produced up to 366–905 labels per node, and 826 MB at n = 58 in Python. A C version would store 24 bytes per label instead of a Python object, roughly 30× less.

**Lower-memory variants.** I've added these to the plan as a new section, "3.4a Memory":

1. **Exact optimum under a memory cap (recommended).** Discard labels whose predicted per-step size exceeds a cap. Because the discard rule is monotone in the predicted state count, the answer is still the exact minimum work subject to the cap. It also prunes the labels that make the fronts large, so the optimiser uses less memory too. The model's absolute predictions are off by 1–2 orders of magnitude, so the cap should be relative:
   - first find the smallest achievable predicted peak B with a bottleneck pass over the DAG;
   - then allow r·B, set per machine with `--kpw-mem-slack r` (no cap by default).
2. **A spill penalty** in the cost, like your ext factor. It is also exact, but has the same calibration problem, so I'd add it only after refitting the model.
3. **Bounded labels** (`--kpw-max-labels L`), which `orders.optimize` already supports. The result may be suboptimal, the optimiser's memory is at most DAG size × L, and it reports whether truncation happened.

I'd skip the second greedy rule (`-g`) for this. It shrinks the DAG 2–6×, but your earlier measurements showed its orderings are noticeably worse.

With your decisions folded in (old orderings kept behind an option,
the C optimiser, the fitted model as default), the plan is ready to
implement. Should the memory cap go in from the start, or after a
first version without it?

Question
========
Run a first version without the memory cap.

Answer
======
I'll implement the first version: the kpw ordering path in `ham_count`, the fitted work model, caching, tests, and an end-to-end run. The optimiser stays in Python for now; the C optimiser is a separate change to kobayashi-pw. First, the repo layout:

I have what I need. Writing the fitted work model module first:

Now `ham_kpw.py`: building and caching the DAG, and the exact optima.

Now the `ham_count.py` changes:

Now fitting `work_model.json` from the Linux logs through the new CLI path:

Now the tests:

Square-sum graphs have too many orderings to enumerate, except for n ≤ 12. I'll enumerate on those plus small random graphs:

All pass; the six skips are the MaxSAT comparisons, because `separation` isn't installed here. I'll try installing it from GitHub:

All 50 tests pass. Now end-to-end runs through `ham_count`, starting with the old MaxSAT path to check it is unchanged:

Now n = 57 with the sort-merge backend, run twice to exercise the cache, plus the candidate-reranking option at n = 50:

The main path works, and the cached second run skips kpw. The candidate reranking crashed inside the existing `partial_dp_time_c`:

That's a pre-existing bug: `partial_dp_time_c` predates the cycle-count outputs added to the C function, so `--multi-start-validate` is broken for MaxSAT too. Let me see how the full count calls it:

Works. A quick check of `--kpw-polish` and of setting `ordering: kpw` in a config file:

Everything works end to end. Now `pyproject.toml`:

Adding an "Implemented" section to the plan:

The first version is written into your repo and working. `uv run count 57 --ordering kpw --sort-merge` gets its ordering from the kobayashi-pw DAG and the exact optimum of the fitted work model over it, with no MaxSAT and no annealing. Run `uv sync` first: `pyproject.toml` now has kobayashi-pw as an editable path dependency (`../kobayashi-pw`), and `uv.lock` needs updating.

All 50 new tests pass, including the pathwidth comparison against MaxSAT for n = 15–40. Every DP count I ran matched `data/b071983.txt`.

- **Speed:** at n = 57 the ordering takes 17.6 s, against 67 s for MaxSAT + SA in the old log. 16 s of it is the Python optimiser, which the C optimiser will remove.
- **Caching:** the DAG is saved in `data/dags/` the first time, so later runs on the same graph skip `kpw`.
- **Old orderings:** `--ordering maxsat` is still the default and behaves as before. `--ordering bfs` gives the BFS ordering, and `--no-pw` still works as an alias for it.

**What's new:**
- **`ham_kpw.py`** builds and caches the DAG and minimises the chosen cost over it: `work` (the default), `v7` or `v2`. Every ordering it returns is checked to be pathwidth-optimal.
- **`work_model.py`** holds the fitted model and a `fit-work-model` command for refitting from profile logs. It reads the ordering from each log's `vertex` column, skips logs that aren't G(n) runs, and uses each distinct ordering once. `work_model.json` is fitted on the 19 distinct Linux runs (n = 57–75). New runs made with kpw orderings can be added when you refit; that is the guard against the model being gamed.
- **New `ham_count` options:**
  - `--kpw-cost` and `--work-model` choose the cost;
  - `--kpw-bin`, `--kpw-threads`, `--dag-cache` and `--kpw-greedy2` control building the DAG;
  - `--kpw-max-labels L` bounds the optimiser's memory, at the risk of a suboptimal ordering;
  - `--kpw-polish` runs the old annealing starting from the DAG optimum;
  - `--kpw-candidates K` picks among several DAG orderings by a partial DP.

  The result line is unchanged, so anything that parses the logs still works.

**A bug I fixed along the way:** `partial_dp_time_c` in `ham_dp_c.py` was passing 12 arguments to a C function that has taken 14 since cycle counting was added. So `--multi-start-validate` has been raising a `TypeError` with the MaxSAT path too.

When `--kpw-candidates` ranks candidates, each one prints `sa_cost=0.000e+00`, because the existing ranking routine expects an annealing cost. The chosen candidate is named on the next line.

The C optimiser for kobayashi-pw is next. It's a change to that repo,
so I'll write its plan first.

Question
========

Answer
======

I'll start by reading the parts of kobayashi-pw the C optimiser has to reproduce exactly: the expansion routine (its order determines each block's step features) and the C programs' DAG and bitset code.

Now the Python side in kobayashi-pw: the shared work features, a matching log-linear model, and `optimize_file`, which runs `kopt`.

I wrote a garbled body for `optimize_file`: it runs `kopt` twice and passes `-q`, which suppresses the ordering. Rewriting it cleanly:

Compiles without warnings. Checking C against Python on the same DAG files for all three models:

Now `work_model.py` takes its features from kobayashi-pw, so `fit-work-model`, the Python optimiser and `kopt` all use one definition:

Updating the hamiltonian tests for the new handle and adding a C-vs-Python check:

Now a test file for kobayashi-pw itself:

All 16 kobayashi-pw tests pass, including the three new `kopt` tests against Python and brute-force enumeration. Now the Makefile, `.gitignore`, README and change notes:

The repo's active changelog is the root `CHANGES.md`, which you have uncommitted edits in. Rather than touch it, I'll put the `kopt` notes in a new `doc/kopt.md`:

Now the plan document's new section, including the extendible-hashing vs sort/merge answer:

The Makefile is marked protected for remote writes, so `make kopt` will need one rule added by hand. Let me check that the rest landed and builds on your machine's toolchain:

The C optimiser is written and tested, and `ham_count --ordering kpw` now uses it. At n = 75 the whole ordering step takes 11 s: 2.9 s for `kpw2` and 8.3 s for the optimiser, using 411 MB. The MaxSAT + annealing ordering took 2179 s. One thing is left for you: the remote tools couldn't write to the kobayashi-pw `Makefile`, so please add this rule to `src/C/Makefile` (and `kopt` to `all`):

```make
kopt: kopt.c
	$(CC) -O3 -march=native -Wall -Wextra -o $@ $< -lm
```

Then run `make kopt`. Without the binary, `ham_count` falls back to the Python optimiser, which takes minutes for n ≥ 59.

**What's new in kobayashi-pw** (`kopt.c` builds cleanly with gcc 11 on your connected machine):
- **`src/C/kopt.c`** runs the same exact optimisation as the Python version on a DAG file, for the V2, V7 and work models.
- **Agreement:** on n = 40–57 it returns exactly the same orderings as Python, for all three costs. It also matches brute-force enumeration of every DAG path on random graphs, with and without `-g`.
- **Speed:**

  | n | ordering time (kpw + kopt) | before |
  |---|---|---|
  | 57 | 0.4 s | 17.6 s with the Python optimiser |
  | 61 | 15.7 s (616 MB) | 930 s with MaxSAT + SA |
  | 75 | 11.3 s (411 MB) | 2179 s with MaxSAT + SA |

- **`orders.py`** now defines the work model's features in one place, so the fit, the Python optimiser and `kopt` all use the same definition. It also adds `optimize_file`, which runs `kopt`. The new `tests/test_kopt.py` and the existing tests all pass (16 in all).
- **Notes:** I put these in `doc/kopt.md` and a new README section, and left the root `CHANGES.md` alone because it has your uncommitted edits.

**Changes in the hamiltonian repo:**
- `ham_kpw` uses `kopt` automatically when it's built, and now reads only the DAG file's header instead of loading the whole DAG into Python.
- New options: `--kpw-python` forces the Python optimiser, and `--kopt-bin` gives an explicit path to the binary.
- All 59 tests pass, including a new one checking that the C and Python optimisers agree.

**Generating the model file:** as you found in the plan, the command is

```
uv run fit-work-model data/linux/profile_*.txt -o hamiltonian/claude/work_model.json
```

Any `ham_count` log with the per-step table works (`--profile` or `-v`). Logs from other graphs, and repeats of an ordering already seen, are skipped.

**Does the model depend on extendible hashing vs sort/merge? No.** It predicts how many states each step produces, and those states are determined by the graph and the ordering alone. Both backends build the same table, so logs from either backend or either machine are valid training data. The quantity being minimised, total work W = Σ states × 2^(back-edges), is the same for both.

What differs is how work turns into time and memory:
- **Extendible hashing:** time is roughly proportional to W, and memory to the table size.
- **Sort/merge:** time is also roughly proportional to W, but steps that spill to SSD cost 2–3× more.

So backend-specific behaviour belongs in the cost being minimised (a spill penalty for sort/merge, or a state cap for extendible hashing), not in the model. That is the memory-cap work deferred from this version.

Question
========
If I put kobayashi-pw on a public github and add it as a dependency
that way, how does the referring repo build the compiled programs? Is
there some other entry in kobayshi-pw that tells the installing
program how to run the Makefile?

Answer
======
Not automatically. Nothing in a `pyproject.toml` says "run this Makefile". The only hook an installer calls is the build backend named in `[build-system]`, and kobayashi-pw uses `uv_build`, which packages Python files and never runs a C compiler. Two consequences:

- **A git dependency gives you only the Python package.** uv builds a wheel from `src/kobayashi_pw` alone. `src/C` sits outside the package, so neither the sources nor any binaries would be installed.
- **It only works now because of the editable path install.** With `path = "../kobayashi-pw", editable = true`, `find_kpw` and `find_kopt` find the binaries next to your checkout, where you ran `make`.

There are three ways to make an installed copy have working programs.

1. **Compile on first use (my recommendation).** Ship the C sources inside the package and have `find_kpw`/`find_kopt` compile them with `cc` into a cache directory the first time they're needed. This is the pattern your DSL already uses, compiling generated C into `~/.cache/kobayashi_pw`.
   - It needs no build-backend change, works the same for git, path and PyPI installs, and compiles with `-march=native` on the machine that runs it. That suits the 64-core box.
   - It needs a C compiler wherever it runs.
   - The sources must be inside the installed package. You can keep them in `src/C` (your convention) if the build backend maps them into the wheel. I know hatchling can do that (`[tool.hatch.build.targets.wheel.force-include]`, e.g. `"src/C" = "kobayashi_pw/csrc"`). I'm not sure `uv_build` can, so that would need checking.

2. **Compile at install time.** Switch to a build backend that runs code while building the wheel:
   - a hatchling build hook (a `hatch_build.py` that runs `make`);
   - setuptools with a custom build step;
   - scikit-build-core (CMake) or meson-python.

   It installs the executables into the package directory. uv then compiles while building the wheel from the git source, which also needs a compiler there. The wheel becomes platform-specific, and if you ever upload wheels you'd have to drop `-march=native`.

3. **Prebuilt wheels.** Build per-platform wheels with cibuildwheel and publish them to PyPI. Users then need no compiler, but this only makes sense if you want it on PyPI, and it gives up `-march=native`.

Option 1 changes the least and fits how you already work. Its cost is
roughly 1–2 s of compilation on the first call per machine and binary,
with the cache keyed on a hash of the source and compiler flags. If
you'd like it, I'll write the plan first, covering the file layout,
cache location, invalidation, and how the hamiltonian repo's
dependency line would change.

Question
========
I like option 1: compile on first use. An interesting alternative is
to use the DSL in kobayashi-pw. Can you write the DAG optimization in
terms of that DSL, or do you need to add additional top-level specs in
the DSL design?

Answer
======
To answer that I need to read the DSL's design and code.

Not with the DSL as it stands. The move system is already there, but the objective, the engine and part of the IR would be new, so it needs additions to the design. I read `doc/dsl_design.md`, `dsl_milestone1.md` and the DSL code (`specs.py`, `problem.py`, `ir.py`, `eager.py`, `dsl_engine.h`) to check this.

**What already fits.** The DAG optimisation runs over exactly the move system that `specs.vertex_separation()` defines:
- the states are relevant sets;
- the moves are x followed by its closure;
- `apply` returns the full added block, as obligation O4 requires.

`kopt` recomputes exactly those arcs and blocks. So the spec itself (closure rules, moves, key) would not change.

**What is missing:**

1. **A different objective and a second engine.** The DSL compiles specs against one fixed engine, E, which minimises the largest step cost along a path. The DAG problem is a dynamic program over a fixed DAG with a path-dependent objective: a running cost plus a carried state count p, which needs a set of Pareto labels at each node. That is a second fixed engine (say `dsl_dagopt.h`, the same algorithm as `kopt.c`) with its own theorem:
   - when e = 0 (additive costs), one label per node suffices;
   - when e ≥ 0, the cost-to-go is nondecreasing in p, so keeping the Pareto labels is exact;
   - truncating labels gives a flagged, possibly suboptimal answer.

   The design already lists `output dag(complete, prune=co-reachable)`, `pareto(secondary)` and `k-best` as Layer-D outputs, but they are milestone-2 items. The DAG output isn't implemented in the DSL yet.

2. **A new top-level clause in Layer P**, something like
   ```
   secondary  step(T, v) = (a, e, k)   direction reverse   by pareto-monotone | additive
   ```
   - **Direction.** The cost is evaluated in the frontier-DP direction: T = V∖S is the placed prefix and the block is reversed. That is expressible with the complement, which the IR has.
   - **Derived arc value.** It would be a fold of (a, e, k) along the reversed block. This is the analogue of the derived arc cost, which today is the maximum of w along the block.
   - **Lemma library.** Two new rows: `additive` and `pareto-monotone` (the second with hypothesis e ≥ 0).

3. **IR extensions.** The IR has only Set, Int, Bool, Vertex and Seq types. The work model and V7 also need:
   - a real-number type with exp, log and powers;
   - a components primitive, such as `sum over components of G[A] of g(size)`, for the Bell-number term;
   - table constants (the Bell numbers) and integer-to-real casts.

   Everything else in the work model is set algebra the IR already has. For example, F(T) = T ∩ N(V∖T); n_back = |N(v) ∩ T ∩ F′|; e_bag = Σ_{u∈F′} |N(u) ∩ F′| / 2.

4. **Block order becomes part of the semantics.** For the bottleneck search, the order of closure additions within a block doesn't matter: closure steps never raise the cost, and the closed set is the same in any order. For secondary costs the order matters, because the features are computed step by step. The DSL's compiled closure may use a neighbour worklist, whose order differs from `core.expand`'s sweep, and that would give different blocks and a different optimum. The clean rule: when a `secondary` clause is present, closure is the sweep, as in eager mode, `core.expand` and `kopt`.

**What it would buy:**
- **Models become Python.** The work model, V7 and V2 would each be about 15 lines of spec, traced and compiled, instead of hand-written C in `kopt.c`. A refitted or reshaped model (new features, a memory penalty) would need no C changes.
- **Checking comes for free.** `check()` could compare the compiled and eager modes, and both against brute-force enumeration of DAG paths, the way `test_kopt.py` does now.
- **Compile-on-first-use is already built.** The DSL's JIT already compiles with a lock and atomic rename, caches the result, keys the cache on hashes of the IR and compiler, and handles W buckets. Option 1 for `kpw`/`kpw2`/`kopt` can reuse that machinery.

**The cost:** the milestone-2 DAG output, the real-number IR with the components primitive, the second engine and the two lemmas. `kopt` is effectively the hand-written reference that the generated code would have to match, in the way `kpw`/`kcw` served for milestone 1.

So the DAG optimisation is a modest extension of the DSL rather than
something it can express today. Since this changes the design, I'd
write it up first as an addendum to `dsl_design.md`, with the clause
syntax, the IR additions, the engine and its theorem, the block-order
rule, and the acceptance tests against `kopt`. Should I do that,
together with the plan for compile-on-first-use?

Question
========
Write up the design, and I'll look it over and tell you to proceed.

Answer
======
I'll check the JIT and IR details the design will build on, so the write-up refers to what is actually there.

I've written the design as `kobayashi-pw/doc/dsl_secondary_design.md`. Nothing is implemented yet.

**Part A: secondary costs in the DSL.**
- **New `secondary` clause.** The step model returns (a, e, k) from the placed prefix T and the vertex v. Its fold along each block is derived automatically, the way the bottleneck arc cost already is. Several named costs (V2, V7, work) can share one move system and one DAG.
- **New engine with an exactness theorem.** A fixed second engine runs `kopt`'s Pareto-label DP, alongside the existing bottleneck engine. Two new library lemmas cover it: `additive` and `pareto_monotone`, the second with hypotheses e ≥ 0 and e·k ≥ 0. A sign analysis on the expressions confirms both hypotheses for V2, V7 and the work model.
- **Block order is pinned.** Once a secondary cost is present, the order of vertices within a block affects the answer. So the closure must be the sweep used by `core.expand` and `kopt`, not the DSL's faster neighbour-worklist order.
- **IR additions:**
  - a real-number type with exp, log and powers;
  - constant tables (log Bell numbers);
  - a sum over connected components;
  - runtime parameters, so refitting `work_model.json` doesn't force a recompile.
- **The DAG itself.** The DSL engine gains a complete mode with backward pruning, which is the existing milestone-2 `output dag` item. It can also read the DAG files `kpw2` writes.
- **Acceptance:**
  - brute-force enumeration of all DAG paths for n ≤ 12;
  - compiled and eager modes agree;
  - identical orderings to `kopt` on the square-sum graphs n = 40–75;
  - within 1.3× of `kopt`'s time.

**Part B: compile on first use.**
- The C sources ship inside the package.
- A new `kobayashi_pw/native.py` compiles `kpw`, `kpw2`, `kopt` and the rest on demand, with `-march=native`. Binaries are cached under `$KPW_CACHE`, keyed on the source, flags and compiler, with locking. It reuses the DSL's existing JIT helpers.
- The word size W is chosen from n, which removes the n ≤ 128 limit of today's `kpw2` build.
- Every place that currently looks for binaries next to the source tree switches to `native.py`, in both kobayashi-pw and hamiltonian. Hamiltonian can then depend on kobayashi-pw from git.

I propose doing Part B first: it's small and independent, and it's what makes the git dependency work.

The document ends with three questions for you:
1. **Build backend.** Is switching to hatchling acceptable if `uv_build` can't include `src/C`? I'll check whether it can before changing anything. The alternative is moving the C sources under the package directory.
2. **Where the work-model spec lives.** I lean to keeping V2 and V7 in kobayashi-pw, where they already are, and putting the work model in hamiltonian.
3. **`kopt.c` after acceptance.** Keep it as the hand-written reference, or retire it?

Question
========
1: yes hatchling is good. 2: that sounds ok, since, in the future I
might use kobayashi-pw in other optimizations. 3: Keep it as a
hand-written reference. The last time I looked the hand-written kpw
was slightly better than the output of the DSL. If this is the case
with kopt we should keep it around to analyze it and see if we can
make the DSL output match it.

Answer
======
I'll record your three decisions in the design document so it stays the reference for implementation.

I've replaced the open questions at the end of `doc/dsl_secondary_design.md` with a "Decisions" section recording your three answers:

1. **Build backend:** hatchling, with `src/C` packaged inside the wheel as `kobayashi_pw/_csrc`.
2. **Where the models live:** V2 and V7 stay in kobayashi-pw. The work model's features, fit and spec stay in hamiltonian, which passes the coefficients to the DSL as runtime parameters. kobayashi-pw keeps only the general machinery, so it stays reusable for other optimisations.
3. **`kopt.c`** stays as the hand-written reference. I changed the speed acceptance test from a pass/fail threshold to a measurement. If the generated optimiser is slower than `kopt`, I'll profile both and write up the specific differences, with the code-generation changes that would close them, in `doc/dsl_codegen_perf.md` before the DSL path becomes the default in hamiltonian.

When you've finished looking it over and want me to proceed, I'll
start with Part B, compile-on-first-use, then Part A in the order
given in §B7.

Question
========
Do part B first, then part A.

Answer
======
Starting Part B. I need a complete, current copy of kobayashi-pw (you have uncommitted changes in several of the files involved), so I'll list what's there.

`kpw2` and `kopt` take `-DW`; the others have fixed sizes. Writing `native.py`:

Now the call sites: `orders_cli`, `orders` and the DSL's `jit.py`.

Now packaging, with hatchling and an explicit file list so stray binaries in `src/C` can't end up in the wheel:

`build.toml` at the root is an unused leftover holding the old `uv_build` stanza; I'll leave it and mention it. Now the tests: point the existing ones at `native.py`, and add `tests/test_native.py`.

All 63 kobayashi-pw tests pass, with no prebuilt binaries in `src/C`. Now the real test of Part B: build a wheel, install it into an empty virtual environment, and run from outside the source tree.

All 59 hamiltonian tests pass with binaries coming from `native.py`. Now the kobayashi-pw README, and a status note in the design document:

Copying Part B back. First a check that your hamiltonian files are still the versions I last wrote:

Part B is done and copied back. A short progress note before Part A, which is much larger:

**Part B (compile on first use) is done and copied back to both repos.**

- **What changed:**
  - kobayashi-pw now builds with hatchling, and the wheel carries the C sources as `kobayashi_pw/_csrc`;
  - the new `kobayashi_pw/native.py` compiles kpw, kpw2, kopt, kcw, bab and fullcount on first use, with `-march=native`, cached under `~/.cache/kobayashi_pw/bin`;
  - kpw2 and kopt get the word count W that n needs, so there is no n ≤ 128 limit;
  - `orders`, `orders_cli`, the DSL JIT and hamiltonian's `ham_kpw` all get their binaries from it;
  - there is a new `kobayashi-pw-build` command for compiling ahead of time.
- **Tests:** 63 kobayashi-pw tests (including the new `tests/test_native.py`) and 59 hamiltonian tests pass.
- **Wheel check:** I built a wheel, installed it into an empty virtual environment, and ran from outside the source tree. kpw (n = 63), kpw2 (n = 75), kopt and a DSL solve all compiled and ran from the packaged sources.
- **What you need to do:** run `uv sync` in kobayashi-pw, because the build backend changed. In hamiltonian the dependency is still the editable path; a pinned git line is ready, commented out, for when the repo is public.

Starting Part A now.

The IR traces as intended. Now eager evaluation of the new nodes:

Now the lemma library. Checking how lemmas are declared:

Now `Problem.secondary`/`param`/`table`, and the eager E2 with complete-mode search:

Now the eager engine: complete mode, E2, and the step evaluation shared by the fold and by whole-ordering evaluation.

Now the hampath V2/V7 models as DSL specs (they stay in kobayashi-pw), plus a reusable log-linear work form for hamiltonian:

Before the C side, checking the eager E2 against `orders.optimize` on small square-sum graphs, for all three costs:

The eager E2 matches `orders.optimize` exactly, including nontrivial Pareto fronts (up to 10 labels per node at n = 30). Now the compiled side. Here is the full engine loop I'm about to refactor:

The refactor keeps all 11 existing DSL tests passing. Now code generation for real expressions and the secondary functions:

Now the generated `dag_arcs`, the per-cost `sec_step_*`, and `sec_block`:

Now the E2 engine header, the C counterpart of the eager `dagopt` and of `kopt`:

Now the JIT side: include `dsl_dagopt.h` when secondary clauses are present, and a `run_dagopt` binding.

Identical orderings at every n so far, with costs equal to about 10⁻¹⁵, and the DSL total time is within about 10% of `kpw` + `kopt`. Continuing to n = 75:

Still identical to eager mode and to `kopt` (9 of 9 cases). Re-measuring the timings:

V7's density term goes through an integer `//` (e_bag = Σ … // 2), which the sign analysis didn't cover. Adding it:

The wheel-list test caught the new `dsl_dagopt.h` header, as intended. Adding it to the wheel:

All 81 kobayashi-pw tests pass. Now hamiltonian: the DSL path behind an option (`--kpw-engine dsl`), with `kopt` staying the default, as agreed.

Now a short section in the hamiltonian plan document, then copying everything back:

Checking that the files I last wrote are unchanged on your side before overwriting:

All match. Copying Part A back, with modification-time guards on the files I hadn't touched before:

Part A is finished, so both parts are done, Part B first as you asked. All the files are now in both repos on your machine. Nothing was rejected: the files you could have edited were only overwritten after checking they hadn't changed since I last wrote them. Run `uv sync` in kobayashi-pw to pick up the new C header. That header is `dsl_dagopt.h`, and it's now in the wheel's list of C sources.

**What Part A adds.** The DSL can now express secondary costs, i.e. the optimisation over the DAG of optimal orderings:
- **Declaring a cost:** `P.secondary(step, name, direction="reverse")` declares one, with `table()` and `param(name, size, nonneg)` as inputs to it.
- **Solving:** `solve(output="pareto", secondary=..., params=...)` runs a Pareto-label DP over the complete relevant-set table. It uses either the eager interpreter or generated C that is compiled on first use.
- **The work model:** `dsl/hampath.py` writes your V2, V7 and fitted log-linear model as DSL specs.
- **Exactness:** the lemma checker has two new checks, `additive` and `pareto_monotone`. `pareto_monotone` is the condition of Theorem E2 (e ≥ 0 and e·k ≥ 0). When it holds, a result marked `exact=True` is the proven optimum, unless the `max_labels` cap truncated the label sets, in which case `exact` is false.

**Correctness.** This was checked empirically on the cases tested, not proven in general:
- The eager DP agrees with brute force over every optimal path, across 180 (graph, cost) pairs.
- The compiled code agrees with eager, and both agree with `orders.optimize`.
- On G₄₀–G₇₅ the DSL picks exactly the same ordering as kopt, with costs equal to within about 1e‑15 relative.

**Speed.** The DSL version takes 0.69–1.02× kopt's time on the optimisation itself, and 0.66–1.02× of kpw `-c` plus kopt end to end. My first measurement had it 1.05–1.8× slower, but that was a mistake: I was counting the pruning of dead states as part of the DSL's optimisation, whereas kpw counts it under `-c`. The prefix-union codegen change I tried made no measurable difference. `dsl_codegen_perf.md` has the details. So unlike last time with kpw, the DSL output doesn't trail the hand-written code here; kopt stays as the reference anyway.

**In hamiltonian.** `ham_count --kpw-engine dsl` is available, and kopt is still the default. A new test checks that the DSL engine matches kopt at n=30, 40 and 50. An end-to-end run at n=57 with the DSL engine gave the correct count, and the ordering step took 0.39 s.

**Where it departs from the design** (listed in `dsl_secondary_design.md`):
- Only the reverse direction is implemented; `direction="forward"` raises `NotImplementedError`.
- The DSL has no k-best or sampling of near-optimal orderings. `--kpw-candidates` still goes through the `KPWDAG3` file.
- The greedy2 block order only matches kopt when the graph is treated as one component, so `_solve_pareto` doesn't split it into components.

Test status: 81 pass in kobayashi-pw and 68 in hamiltonian. I ran them in my copies here, not on your machine, so the suites should be run on your side after `uv sync`. Nothing has been committed to git, so the changes are ready for you to review and commit.

Files are in:
- **kobayashi-pw:**
  - `src/kobayashi_pw/dsl/`: `ir.py`, `eager.py`, `lemma.py`, `problem.py`, `codegen.py`, `jit.py`, new `hampath.py`
  - `src/C/`: `dsl_rt.h`, `dsl_engine.h`, new `dsl_dagopt.h`
  - `tests/test_dsl_secondary.py` (new)
  - `pyproject.toml`, `README.md`
  - `doc/dsl_codegen_perf.md`, `doc/dsl_secondary_design.md`
- **hamiltonian:**
  - `hamiltonian/claude/ham_kpw.py`, `ham_count.py`
  - `tests/test_kpw.py`
  - `doc/kpw_ordering_plan.md`, which gains a section 7
  
  
Question
========
The progam `ham_dp_c` uses the values in `machine.yaml` to optimize the
use of cache. However, `machine.yaml` does not have information about
the size of the cache lines, nor the set associativity of the various
Lx caches. Would taking account of this information help to make more
efficient use of memory? If so, can you work on changes which will
both gather that information (I understand that `sysctl` on the Mac
does not contain information about associcativity, but that must be
determined via experimentation) and use it in `ham_dp_c`. This may be
especially significant with respect to parallel threads so that they
don't "step on each other".

Answer
======
I'll start by reading how `ham_dp_c` currently uses `machine.yaml` and where its memory traffic goes. Then I'll write a design doc for you to review before I change any code.

Yes, in a few places, but these changes are not where most of the time goes at n=75. I've written the analysis and plan to `doc/cache_geometry_plan.md` in the hamiltonian repo. No code is changed yet.

**Your logs show where the time goes.** Breaking the instrumentation lines into phases:
- **Linux, n=75 (`profile_75log.txt`):**
  - **51.7% (3026 s of 5851 s) is spent in the per-worker `sm_merge_runs` loop.** In RAM mode it runs one worker after another on the main thread (around line 3033). The log already labels it `merge_runs(seq)`, and it accounts for the unexplained 60–70% of every large RAM-mode step.
  - Ext-mode steps already merge all runs in one parallel pass and don't have this gap.
  - Replacing it with one parallel merge over all the runs, which is what the ext path already does, should cut the total roughly 40–50%. That's an estimate from the phase table, not a measurement. Peak memory would be no higher than now.
- **M2 Pro, n=61:** the radix sort (the "flush" phase) is 31% of the time. That is where cache geometry genuinely pays.
- **Compute phase:** about 120 ns per subset evaluation at only about 2 GB/s across all threads. It is limited by the arithmetic, so cache geometry won't help it.

**Where the geometry matters:**
- **Associativity matters essentially only for L1.** L1 is indexed by virtual address bits inside the page, so power-of-two strides alias predictably there.
  - For L2 and L3, the index bits above the page come from the physical page number, which is effectively random with 4 KiB or 16 KiB pages. Layout can neither cause nor fix conflicts there unless you use huge pages. Zen 4's L3 and Apple's SLC are also hashed.
  - I'd still measure L2 associativity on the Mac, using minimal eviction sets (Vila–Köpf–Morales), which works despite the random mapping. But the result is mostly a sanity check, not a tuning knob.
- **Sharing matters more for threads stepping on each other.** Several buffer sizes assume each thread owns a whole cache. In fact the Apple L2 is shared by a whole cluster, the Zen 4 L3 is shared per CCD, and SMT siblings would share L1.
- **Existing false-sharing padding doesn't work.** The per-worker structs are allocated with `calloc`, which ignores their 64/128-byte alignment attribute.
- **Radix sort layout:** the current buffers start all 256 bucket heads in a single L1 set on the M3 at the start of every pass. I've proposed small, line-aligned buffers that fit in L1, which is conflict-free by construction. Add whole-line flushes that skip the read before a store, and sort in L2-sized chunks after a first partition pass. That cuts the memory round trips from 8 to 2–3.
- **Inconsistency:** `SMEntry` is 16 bytes, but `ham_dp_c.py` and `tune_params.py` assume 24 bytes. The default sizes are therefore off by 1.5×, and the tuner benchmarks a record of a different size from the one the DP uses.

The suggested order is:
1. the parallel merge;
2. the cheap fixes: alignment, per-thread cache shares, the 16-byte sizes;
3. a `cacheprobe.c` measurement tool plus new `machine.yaml` keys;
4. the radix-sort rework, only if a stand-alone benchmark shows ≥1.5×.

The tuning has to be run on your Mac and the Linux machine themselves. The Linux VM I can reach on your Mac reports no cache sizes, so it's only good for correctness tests.

Questions before I start:
- Can you send `lscpu` and `grep . /sys/devices/system/cpu/cpu0/cache/index*/*` from the Linux machine, and say whether SMT was on? The comment in `n75_linux.yaml` ("2 CCDs", 64 cores) doesn't add up for a Zen 4 part, since each CCD has 8 cores.
- Should `cacheprobe.c` go next to `ham_dp_c.c` (how this repo is organised) or in a new `src/C` (your usual layout)?
- Was the serial merge loop deliberate, for example to stay under a
  memory limit at some step?
  
Question
========
Look at data/linux_arch.txt

Answer
======
The Linux machine is laid out differently from what the configuration assumes. It has two EPYC 9124 sockets with 16 cores each and SMT 2. That is 32 physical cores and 64 logical CPUs, split across two NUMA nodes. I've added a new §0a and a NUMA item G to `doc/cache_geometry_plan.md` to cover this.

What this changes:

- **The 64 workers share cores.** `SM_NTHREADS: 64` is the logical CPU count, although the comment in `n75_linux.yaml` says it is physical cores. Each worker therefore shares its L1 (32 KiB, 8-way, 64 sets) and its L2 (1 MiB, 8-way) with its SMT sibling, e.g. cpus 0 and 32.
  - Each thread's real share is 16 KiB of L1, 512 KiB of L2 and 2 MiB of L3. An L3 is 16 MiB, shared by 4 cores (8 threads), and there are 8 of them.
- **The sibling-collision fix (B3) now applies.** Siblings share an L1 whose set index lies inside the 4 KiB page offset. Their radix buffers sit at the same page offsets, so their hot bucket heads compete for the same sets.
  - The fix is to offset odd siblings' buffers by 2 KiB. It only works fully if threads are pinned, so it is paired with G1.
  - The L1-resident sort buffers (C1) must also be sized to the 16 KiB share.
- **`SM_SLC_BYTES: 64 MiB` matches no cache on this machine.** In the parallel ext merge, each of the 64 merge threads gets 64 MiB of stream buffers against a real share of 2 MiB.
  - So in practice it sets the read-ahead size for the SSD/mmap runs, not a cache budget. That value may still be right for I/O, but it should be tuned under that name rather than derived from a cache size.
- **NUMA placement is now an item.** Nothing in `ham_dp_c.c` pins threads or places memory, so under first-touch a large fraction of DRAM traffic is probably remote. This is an inference I haven't measured.
  - The cheapest test needs no code: run around n=70 once under `numactl --interleave=all` and once under the default.
  - Run `SM_NTHREADS=32` against `64` in the same session. I'd expect the sweep to gain from SMT and the bandwidth-bound radix sort possibly to lose.
- **Associativity is now known from sysfs.** On this machine the probe tool only needs to check sysfs, not measure from scratch. The eviction-set measurement is still needed on the Mac.

None of this changes the top item. On that machine the serial per-worker merge is still about half of the n=75 run time, and parallelising it doesn't depend on the cache layout.

Shall I start there? The open questions are where `cacheprobe.c`
should live, and whether the serial merge loop was deliberate.

Question
========
Put cacheprobe.c in src/C. From what I remember the serial merge may
have been deliberate. At one point it was running out of memory. I
would think a better plan would be to adapt its parallelism to the
memory configuration.

Answer
======
I've rewritten section A of `doc/cache_geometry_plan.md` so the merge's parallelism is set by the memory available, and `cacheprobe.c` is now placed in `src/C`. No code has changed yet.

The serial loop's memory cost is clear from the code. Let R be the total size of all workers' sorted runs. Run one worker at a time, the peak is R plus the largest single worker's runs. Run all workers at once, it is about 2R, which is presumably the OOM you hit. The final merge after it already peaks near 2R unless the pairwise fallback takes over.

The plan has two steps:

- **A0: parallel inside each worker's merge, no extra memory.** `sm_merge_runs` already splits each worker's runs into 256 independent hash buckets. A0 merges those buckets with up to 64 threads, writing each into its upper-bound position in the output and then compacting. Workers stay one at a time, so the peak memory is exactly what it is today. This is the smallest change, and it should take most of the 51.7% the serial loop costs at n=75. That is an estimate, not yet measured.
- **A1: one global parallel merge that frees its input as it goes.** This replaces both merge stages with a single parallel merge over all the runs, the same algorithm the ext path already uses. Each thread hands back the input pages it has consumed (`madvise`), and output only becomes resident as it is written. The peak is then at most R plus the threads' unreleased input, versus about 2R now.
  - **How parallelism follows memory:** from the free memory at merge time, it picks the thread count and how often input is released so the unreleased amount fits in half the free memory. If even one thread won't fit, it falls back to A0. For example, 192 streams, 64 threads and a 256 KiB release step leave about 3 GiB unreleased.
  - **Follow-on gain:** because A1 lowers the peak, the RAM-or-SSD decision could keep some steps in RAM that now spill to SSD. I'd enable that only after A1 is validated.

Both steps log the predicted peak and the real peak RSS for every step, so the memory claim gets measured rather than assumed. I can run the correctness tests up to n=60, checking state counts, per-step checksums and final counts against the current code. The Linux timing run at about n=70 is yours.

Shall I go ahead with A0 and then A1, followed by the cheap fixes (B1,
B2, D) and `src/C/cacheprobe.c`?

Question
========
Do A0 then A1. Then B1, B2, D and src/C/cacheprobe.c

Answer
======
I've implemented A0, A1, B1, B2, D and `src/C/cacheprobe.c`, and all of them are now in the repo on your Mac. In every configuration I tried, each step's state table and the final count match the unmodified code exactly. The memory reduction is measured. I haven't been able to show a speed-up you can rely on, because I could only run on 2-core virtual machines. Nothing is committed to git.

- **A0:** each worker's merge now runs on all threads, split over up to 4096 hash buckets. Workers are still merged one at a time, so peak memory is unchanged.
- **A1:** a single parallel merge over all workers' runs that hands each input page back to the OS once it has been read. It picks its thread count and release size from the memory available, and falls back to A0 when even one thread won't fit.
  - At n=47 (4 M-entry cap, 4 threads), peak memory during the merge dropped from about 2R to about R, where R is the size of the runs being merged:

    | step | R | A0 peak | A1 peak |
    |---|---|---|---|
    | 33 | 0.23 GB | 0.71 GB | 0.45 GB |
    | 34 | 0.34 GB | 1.00 GB | 0.59 GB |
    | 35 | 0.39 GB | 1.07 GB | 0.63 GB |
    | 36 | 0.52 GB | 1.45 GB | 0.65 GB |

  - One change from the plan: each thread writes into its own chunks, and `nxt` is then allocated at its exact size and filled by a parallel copy. That avoids a large virtual reservation (no overcommit needed) and a serial compaction step.
- **B1:** per-thread state is padded to 128 bytes and actually allocated at 128-byte alignment. `calloc` had been ignoring the alignment attribute.
- **B2:** the default `SM_RBUF_SIZE` and `SM_BB_BUF` now use each thread's share of L2 and L1, taking SMT siblings and the Apple cluster L2 into account. Values in `machine.yaml` still override them.
  - Correction to my earlier note: the `SM_BB_BUF` formula was right except that it used the whole L1, not the thread's share. Its P is the number of streams one merge reads, not the number of threads sharing L1.
- **D:** the 24→16-byte entry size is fixed in `ham_dp_c.py` and `tune_params.py`, including the tuner's benchmark struct. macOS cache sizes now come from the P-core (`perflevel0`) keys.
- **`cacheprobe.c`:** `tune_params.py` runs it when writing `machine.yaml`, or alone with `--probe-only`, and it fills new informational `_cache` keys.
  - L1 matched sysfs exactly in the cloud VM.
  - The coherence-granularity and L2-associativity measurements fail in virtual machines. The probe then reports `null` rather than a guess.
  - In the Linux VM on your Mac the L1 result varied from run to run, so treat any VM numbers as untrustworthy. None of these measurements has been tested on real hardware.

Things you should know before running it:
- Merge timings, on 2 vCPUs only: at n=47 the per-worker stage went from 16.4 s to 9.4 s with A0, and A1's single merge took 11.5 s against 20.1 s for the old two stages. The real test is the 32-core Linux machine, where the serial stage was 52% of the n=75 run.
- On Linux the file now defines `_DEFAULT_SOURCE` so that `madvise` is actually available. As a side effect, an existing `MADV_FREE` call in the hash-table (EH) path, which had been silently compiled out on Linux, now runs.
- I left the RAM-or-SSD decision (`sm_should_use_ext`) at its old 2× budget. Lowering it to A1's bound should keep more steps in RAM, but I'd do that only after you've validated A1 on the Linux machine.
- `count_hamiltonian_paths_sm` gained a `ram_bytes=` argument so the new test doesn't depend on the host's RAM. The test also passes on the aarch64 Linux VM.
- The new environment switches (`HAM_SM_CKSUM`, `HAM_SM_MERGE`, `HAM_SM_MERGE_BUDGET`) are for testing and change nothing by default.

To check on your machines:
1. Run `uv run pytest tests/test_sm_merge.py`.
2. Run `uv run python -m hamiltonian.claude.tune_params --probe-only --output hamiltonian/claude/machine.yaml` natively on the Mac and on the Linux machine.
3. Do an n≈70 run on the Linux machine with `instrument: true`. The new `[inst] ram_merge:` lines show the chosen thread count and release size, plus the predicted and measured peak for each step. Setting `HAM_SM_MERGE=a0` gives an A/B comparison against A0.

Section 6 of `cache_geometry_plan.md` has the full details.

Files are in `~/Programming/Combinatorics/hamiltonian`:
- `hamiltonian/claude/ham_dp_c.c`
- `hamiltonian/claude/ham_dp_c.py`
- `hamiltonian/claude/tune_params.py`
- `src/C/cacheprobe.c`
- `tests/test_sm_merge.py`
- `doc/cache_geometry_plan.md`

Question
========
I ran tune_params on the linux machine and ran n=69 with the generated
machine.yaml. It now took 410.119 seconds for n=69.

Answer
======
That's a large gain: the only earlier n=69 run on that machine took 1465 s of DP time. But the logs I have can't say how much of the 3.6× comes from the new merge code.

That earlier run is `data/linux/profile_69linux.txt`, from April 12. It differs from today's run in three ways besides the code changes:

- **Ordering.** April used the MaxSAT ordering (pathwidth 14, profile 805). Your recent runs use the kpw work-model ordering, which by itself could account for a large factor.
- **Parameters.** The `machine.yaml` that tune_params just generated is different from the hand-tuned one.
- **Code.** Besides A0/A1/B1/B2, the code has changed since April 12. The n=75 log already shows a different instrumentation layout.

A0 probably did nothing on this run. In April every worker produced a single run at every step, so the serial per-worker merge it parallelises had no work to do. A1 replaces the final merge, which was only 15% of that run. Much of the gap was in time the April instrumentation doesn't attribute: about half of each large step.

Two things would settle it:

1. **The new log and the generated `machine.yaml`** (drop them in `data/linux/`). From the `[inst]` lines I can split the 410 s by phase and compare with April step by step. The new `ram_merge:` lines also show the thread count, release size and predicted vs. measured peak memory on the real machine. Please also confirm the count came out as 6,697,388,275,608,246.
2. **One A/B run.** Same ordering, same `machine.yaml`, with
   `HAM_SM_MERGE=a0`. When each worker has a single run, the `a0` path
   is exactly the old merge, so the difference isolates A1. If you
   also want the tuning effect, run the old hand-tuned yaml with the
   default merge.
   
Question
========
You can find `new_74_sm_new.txt linux.yaml new_69_sm_new.txt
linux_work_model.json` in `data/linux` I didn't run them
instrumented. Notice that 74 now dies at the very last step because of
overflow 64 bits.

Answer
======
The overflow looks real, and the run already contains a cheap check of that. I've written up a diagnosis and a proposed fix in `doc/sm_overflow_plan.md`; nothing is implemented yet.

**Why it overflows.** Each state's count is stored as a 64-bit integer, and the comment saying that is safe for n ≤ ~110 is wrong. It reasons from the final answer. But a state's count is the number of partial path systems on the vertices processed so far, and H(n) doesn't bound that. Most of those partial systems are discarded only in the final eliminations.

**Why it shows up now.** The kpw ordering narrows the tail of the run a lot:

| step | states, kpw order (new run) | states, MaxSAT order (April) |
|---|---|---|
| 71 | 4,492 | 431,514 |
| 72 | 57 | 245 |

The same total count is packed into 4× fewer states at step 72, so each state's count is about 4× larger. H(74) ≈ 2.18·10¹⁸ is already within a factor of 8.5 of 2⁶⁴, so that is enough to wrap.

**Why I don't think the new merge caused it.** The new merge never changes a count: in every test the per-step checksums, including the sum of all counts, matched the old code. Overflow is detected in the same place as before. The check only runs after the last step, so "dies at the very last step" means "is noticed then". The wrap itself was probably at steps 71–72.

**The run is not wasted.** The DP only ever adds counts, so with wrapping arithmetic everything it computes is exact modulo 2⁶⁴. Your run computed H(74) mod 2⁶⁴, and the Python code throws that away when it raises the error. If that residue equals 2,176,025,661,130,535,741 (the earlier H(74), which is below 2⁶⁴), it confirms both that the wrap is genuine and that the new merge code is correct.

**What I propose:**
1. On overflow, report the residue mod 2⁶⁴ and the first step at which a count wrapped, instead of discarding the result.
2. For exact results without wider entries, rerun with all counts reduced modulo the prime 2⁶¹−1 and combine the two residues by the Chinese remainder theorem.
   - This leaves memory and speed per step unchanged, but costs a second run when the first one wrapped.
   - It is exact only if X < 2⁶⁴·(2⁶¹−1) ≈ 2¹²⁵. I have no useful bound proving that; simple degree-product bounds for these graphs come out around 2²⁷⁰.
   - So this needs your call: add a third modulus (≈2¹⁸⁶, still unproven but very comfortable, since H(75) ≈ 2⁶⁹), or certify X with a double-precision lane whose error can be rigorously bounded (more memory, or a third run).
3. I'm not proposing 128-bit counts. They cost 1.5× memory and bandwidth on every step, and memory is what limits n ≥ 75.
4. Separately, `tune_params.py` writes YAML aliases (`*id001`) into `machine.yaml`. They're harmless, but I'd turn them off.

On speed, only totals are comparable, because the ordering, parameters and code all changed at once: n=69 took 410 s against 1465 s in April, and the n=74 DP took 733 s against the best April run's 945 s.

Shall I implement 1, 2 and 4, and which way do you want the 2¹²⁵
hypothesis handled?

Question
========
Do 1 2 and 4. The Chinese Remainder sounds like a good idea.  It's too
bad that the overflow happened at the very last step. Didn't we work
out a rigorous upper bound, or is that too loose?

Answer
======
Yes, we did work one out: the Brégman–Minc bound in `hamiltonian/bounds/bregman.py`. It is too loose to estimate H(n), but tight enough to make the CRT rigorous. Items 1, 2 and 4 are done, and the tests pass both here and on the aarch64 Linux VM.

**Why the bound is enough.** For n = 74 it gives about 2^114, while H(74) is about 2^61. The CRT only needs a modulus above the bound. The 64-bit run plus one prime, 2⁶¹−1, gives a modulus of about 2^125. That certifies every n up to 79; two primes reach about n = 104. The code computes the bound from the graph's degrees and uses the fewest primes that clear it, with a one-bit margin. It also asserts that the result is below the bound.

**The overflow at the last step should now be cheap to recover.** While the DP runs, it keeps a copy of the most recent state table that fits in min(1 GiB, RAM/64), until the first wrap. Any copy taken at or before the wrap step still has exact counts. So after the normal run, only the steps from that copy onward are rerun mod the prime. At n = 74 those last steps have between about 10⁴ and 10⁷ states, so the extra work should be seconds rather than another 733 s. That's an estimate, since I don't know your exact wrap step. If a wrap ever happens at a step too big to copy, the whole DP is rerun once per prime instead. Either way, stderr gets a one-line note saying what was done.

**What changed:**
- **Item 1:** `--on-overflow residue` stops with an `SMOverflow` exception that carries the count mod 2⁶⁴ and the first step that wrapped. `crt`, the default, recovers the exact count.
- **Item 2:** the exact recovery above. The cycle count gets the same treatment.
- **Item 4:** the wrong "safe for n ≤ ~110" comment is replaced, and `tune_params.py` no longer writes `*id001` aliases into `machine.yaml`.
- **A bug fix:** the overflow flag was never reset, so in a run over several n, every n after the first overflow would have been reported as overflowed.
- **Checks:** the step-by-step state tables are bit-identical to the previous code at n = 45 and 47. The new tests force early wraps by pretending counts are 12, 16 or 20 bits wide at n = 41. They check exact paths and cycles by both routes, the residue mode, combining three moduli, and the SSD-spill path.

The memory cost is the saved copy, at most 1 GiB, plus one copy per small step. Simply rerunning your n = 74 job should give the exact count. Expect 2,176,025,661,130,535,741, and the stderr note will report where the wrap happened.

Files are in `~/Programming/Combinatorics/hamiltonian`:
- `hamiltonian/claude/ham_dp_c.c`
- `hamiltonian/claude/ham_dp_c.py`
- `hamiltonian/claude/ham_count.py`
- `hamiltonian/claude/tune_params.py`
- `tests/test_sm_crt.py`
- `doc/sm_overflow_plan.md`

Question
========
Look at `data/linux/new_74_sm_new_instrumented.txt`

Answer
======
The instrumented run gave the right answer in 684.6 s. Every step stayed in RAM; none spilled to SSD.

**Correction:** the counts first wrapped at step 68, not the last step. The message only prints at the end. That's when the table had 48 M states. Its saved copy was 0.77 GB, under the 1 GiB limit, so only steps 68–73 were replayed mod 2⁶¹−1, which took 5.2 s (0.8% of the run).

**Where the time went, against the best April run.** The April run is `profile_74new.txt` (MaxSAT ordering, old code). Compute and flush are summed per-step maxima across the 64 workers.

| | April | now | per unit of work |
|---|---|---|---|
| total DP | 950 s | 679 s | |
| work W = Σ states_in · 2^n_back | 131.4 G | 108.5 G | ordering: −17% |
| compute (sweep) | 496 s | 295 s | 242 → 174 ns per subset·thread (−28%) |
| flush (radix sort) | 289 s | 198 s | 364 → 258 ns per raw entry·thread (−29%) |
| final merge | 237 s | 118 s | 4.65 → 2.40 ns per raw entry (−48%) |
| unattributed | ~0 | 67 s | |

How I read it:
- **Ordering.** The kpw work ordering cuts the work by 17% (raw output is about the same, 49 G against 51 G entries).
- **Merge.** A1 roughly halves the merge cost per entry. It is the only change aimed at that phase, but I can't separate it cleanly from other differences. In April every worker made a single run, so the serial per-worker merge (A0's target) cost nothing in either run. The gain is in the final merge and in how its output is allocated.
- **Compute and flush.** Both got about 28% faster per unit of work, and none of my changes touch them. Likely causes:
  - the smaller worker buffer that tune_params chose (32 M entries against, I believe, the 128 M in `n75_linux.yaml`), which means fewer page faults on large freshly allocated buffers;
  - the other user who was on the machine in April.

  I can't tell these apart from these two logs.

**Memory.** The merge's peak was at most 56 GB (step 48) against a 752 GB budget. With that much headroom, A1 chose the largest release granule, 64 MB. That is bigger than each thread's slice of a run (about 5 MB), so it never released input early. The peak was therefore about R + the output, roughly 1.3 R. That's the intended behaviour: it only trades speed for memory when memory is tight. The predicted peak of about 2R is a deliberately loose upper bound. A1 will only really be exercised at n = 75.

**What's left.** Compute is 43% of the time and is limited by arithmetic, not memory. Flush is 29% and is the radix sort, which the C redesign in the plan targets. Two cheap experiments would settle the attribution:
1. Same kpw ordering and same new `machine.yaml`, but with `HAM_SM_MERGE=a0`. That reproduces the old merge path and isolates A1.
2. Same ordering, but with the old hand-tuned yaml. That isolates the tuning effect on compute and flush.

The instrumentation line now prints `n_runs=0` under A1, because the
runs are freed before the line is printed. I'll fix that with whatever
comes next.

Question
========
I'm running n=75 now with the new code. It just finished step 43 in
709 seconds up until then.

Answer
======
That's a big improvement. The April n = 75 run (`profile_75log.txt`, MaxSAT ordering) had taken 1,929 s by the end of step 43, so the new code is 2.7× faster to that point.

Step 43 doesn't mark the same amount of work in the two runs, though, because the orderings differ. In April, step 43 was only 33% of the way through the 5,851 s total. The run peaked at 3.8 G states at step 41, and the expensive steps ran from about 37 to 62. Most of the cost, including the RAM steps where the serial merge took about half the time, came after step 43. So I wouldn't extrapolate the total from this yet.

Two things to look for when it finishes:

- **Memory.** In the `[inst] ram_merge:` lines, compare `hwm` with `R` and `budget` on the large steps. At n = 74 A1 never needed to release input, because memory was ample. n = 75 is where the release granule G should shrink, and where the measured peak would show whether A1 stays near R.
- **Overflow.** A wrap is very likely, since H(75) ≈ 6.5·10²⁰ > 2⁶⁴. One prime still certifies it: the Brégman bound is 2^116.3, below the 2^125 modulus. It will be cheap only if the first wrap happens at a step small enough to fit the 1 GiB snapshot (64 M states). Otherwise you'll see "full reruns" in the stderr note, which would double the run time.

If that full rerun would be unwelcome, the snapshot limit can be
raised with `crt_snap_bytes`. For example, 32 GB covers about 2 G
states; RAM is not binding at these sizes. It can't be changed on the
run already in progress, but you can still compare against the April
step table now if you send me the output so far.

Question
========
Look at `data/linux/new_75_sm_new.txt` for the partial run up until
now.

Answer
======
Up to step 44 the run looks good: 980 s against 2,190 s for the April run at the same step. That isn't a like-for-like comparison, because the kpw ordering produces bigger tables. The peak so far is 7.15 G states at step 44, against April's 3.8 G peak for the whole run.

**Breakdown so far.** Of the 980 s, compute is 36%, flush 20% and merge 35%. 97 s is not attributed to any phase. The four steps that spilled to SSD (40, 41, 42 and 44) account for 694 s, or 71%.

**A1 under real memory pressure (step 43).** This is the first step where A1 had to adapt:
- R = 135 GB of runs; it picked a 31 MB release granule and the predicted peak was 272 GB.
- The measured peak was 231 GB, about R plus the output, against a 752 GB budget.
- It still released nothing early. With K = 320 runs, each thread's slice of a run is about 6.6 MB, smaller than the 31 MB granule, so release only happens when a thread finishes. That is consistent with the bound, and memory was nowhere near binding.
- The merge took 18.4 s plus 13.5 s of copy, for 9.1 G raw entries.

**The SSD spills look unnecessary.** The RAM-or-SSD test, `sm_should_use_ext`, decides before the sweep and assumes three things:
- every one of the 2^n_back subsets of back edges is valid;
- every worker fills a whole number of buffers;
- the merge needs 2 × raw.

The actual raw output is about a third of its estimate. For step 40 it estimates 412 GB of runs and an 824 GB merge, so it goes to SSD. What was really needed:

| step | states_in | raw (actual) | RAM path peak: max(sweep, A1 merge) | decision |
|---|---|---|---|---|
| 40 | 3.09 G | 8.5 G (136 GB) | ≈ 248 GB | SSD |
| 41 | 3.88 G | 9.7 G (156 GB) | ≈ 282 GB | SSD |
| 42 | 5.33 G | 7.3 G (117 GB) | ≈ 267 GB | SSD |
| 44 | 6.67 G | 19.4 G (310 GB) | ≈ 481 GB | SSD |

Here the sweep peak is the input table plus 64 GB of worker buffers plus the runs, and the merge peak is about R + N. Every one of these would have fit in 752 GB.

Judging by step 43, which stayed in RAM and took 83 s for 9.1 G raw entries, these four steps would have taken roughly 70–90 s each (step 44 around 170 s) instead of 115–271 s. That would have saved something like 300–350 s already, with more of these steps still to come. This is an extrapolation from one step, not a measurement.

**What I'd propose next:** decide during the sweep, not before it. Every run starts in RAM. A shared counter tracks the bytes of RAM runs, and once a run would push the predicted A1 peak past the budget, that worker writes its remaining runs to SSD.
- The merge then reads a mix of RAM and SSD runs. A1's merge already works on any array it can point to, and the SSD runs are memory-mapped, so it needs only a small generalisation.
- It cannot run out of memory, because it is driven by measured sizes rather than an estimate, and it only spills when memory is genuinely short.
- A step-level estimate based on the previous step's measured growth would be simpler, but it isn't safe: growth varies a lot from step to step.

**About the CRT.** With 7 G states in the middle of the run, the first wrap is likely to come at a step much larger than the 1 GiB snapshot (64 M states). If so, recovery will be a full rerun mod 2⁶¹−1, which roughly doubles the time. On this machine `crt_snap_bytes` could safely be tens of GB. For future runs I'd add a `ham_count` option for it, or raise the default to RAM/16.

Shall I write the design for the in-sweep spill decision, plus the
snapshot option, before implementing?

Question
========
Do the design for in-sweep spill plus snapshot. Right now n=75 just
finished step 49 in time 1723 seconds.

I've written the design to `doc/sm_spill_snapshot_design.md`; nothing is implemented yet. Step 49 at 1,723 s compares with 3,434 s for April's run, about 2× faster at that point.

First, a correction: the "ext" steps never went to SSD. The ext store is anonymous memory, so RAM, or swap if the machine has any; it only falls back to a disk temp file if that memory can't be reserved. What makes ext steps slow is their older merge: about 6–7 ns per raw entry, against 3.5 ns for A1. My conclusion stands, though. Steps 40–44 took the slow path because an estimate made before the sweep was 3× too pessimistic.

**The design:**
1. **One run layout, decided while the sweep runs.** A shared counter tracks the bytes of runs held in RAM. When a worker flushes a run, it keeps it in RAM if the step still fits: input table + worker buffers + snapshot + 4 GiB reserve + the smallest merge plan. Otherwise it writes that one run to real disk. This rule bounds memory by construction, with no estimate of the step's output. The old pre-sweep decision (`sm_should_use_ext`) is retired; the old ext merge stays for one release behind `HAM_SM_LAYOUT=ext` for comparison.
2. **Real disk spill.** Each worker gets one deleted-on-open file in `--spill-dir` (default `$TMPDIR`). A1 learns to read buffered file streams alongside RAM streams, and the page cache is dropped after each read. Spilling happens only when RAM is genuinely short.
3. **Snapshots only where a wrap is possible.** Every count after step s is at most U_s = 2^n_back × (sum of the step's input counts). That sum is exact in 128 bits and costs one add per entry. So if U_s < 2⁶⁴, step s provably cannot wrap.
   - **One anchor snapshot** at the first step where U_s ≥ 2⁶⁴. It's kept in RAM up to `--crt-snap-gb`, by default min(RAM/16, 64 GiB), which is 47 GB on the Linux machine. If bigger, it's written once to disk.
   - **Smaller rolling snapshots** after that, as now, until the first actual wrap.
   - **The cost:** a replay never costs more than a full rerun, and usually much less.
4. **New per-step instrumentation:** RAM run bytes, spilled runs, U_s, the snapshot taken, and A1's RAM and file stream counts. It also fixes the bogus `n_runs=0` in the logs.

There are tests for spill (none, partial, all), for the U_s inequality (exact under the 2^b test hook), and for the CRT with the anchor in RAM, on disk, and absent.

From step 43's rate, the fast path would have saved roughly 300–350 s by step 44, with more on the later big steps; that's an extrapolation from one step. Out of scope: a step whose output table alone doesn't fit in RAM. It would now fail early with a clear message instead of swapping.

Two questions:
- Does the Linux machine have swap, and where does `TMPDIR` point? This matters for the default spill directory, and it might explain April's "running out of storage".
- Should I go ahead with implementing this?

Question
========
Go ahead and implement it. I just checked TMPDIR is not set. The temp files are ending up /tmp . I can see the compiled c files from the run there.

Answer
======
The in-sweep spill and the new CRT snapshots are implemented and in the repo. The tests pass here and on the aarch64 Linux VM. Your n=75 run is unaffected: it is using the library it compiled at start-up.

**The 100% CPU you saw.** In "ext" steps the merge itself runs in parallel. Before it, a serial loop sets up the buffers, and after it, one serial `memmove` packs the threads' outputs together. At step 44 that `memmove` moves about 110 GB, and that is the 100% phase. The new merge has neither; it writes per-thread chunks and copies them into place in parallel. A second serial phase happened on every path: freeing the input table, a single `munmap` of up to ~100 GB, which the kernel does on one core. Tables of 1 GiB or more are now released by all threads first.

**What's new:**
- **Spill decision during the sweep.** Every step keeps its runs in RAM and uses the new merge. When a worker finishes a run, it keeps it only if the step still fits its memory budget. Otherwise the run is appended to that worker's spill file, which is deleted as soon as it's created. The merge reads RAM and spilled runs together.
  - The spill directory is `--spill-dir`, defaulting to `$TMPDIR` and then `/tmp`. If the directory is tmpfs (i.e. RAM), the code prints a warning. Please check whether `/tmp` on that machine is tmpfs.
- **CRT snapshots** follow the rigorous bound from the design. The DP now sums each step's input counts exactly in 128 bits.
  - The anchor snapshot is taken at the first step where a wrap becomes possible. It's kept in RAM up to `--crt-snap-gb`, by default min(RAM/16, 64 GiB), or about 47 GB on that machine. A larger one is written once to the spill directory.
  - After that, rolling snapshots of up to 1 GiB follow until the first wrap.
- **The old merge as fallback is gone.** Even the new merge's smallest plan peaks lower, and the old merge can't read spilled runs. This also fixed a crash: the old fallback aborted on a step with no runs at all.
- **The old layout** is still available with `HAM_SM_LAYOUT=ext`, for comparison.
- **Instrumentation** now shows per step: run counts with spilled runs and bytes, the spill limit, the snapshot taken, and the new merge's RAM and file stream counts. `n_runs` is now the total over all workers.

**Tests:**
- In `tests/test_sm_merge.py`, per-step state checksums are identical across all variants at n = 36 and 41: no spill, partial spill, everything spilled, the old layout, and nominal RAM so low that every run spills.
- `tests/test_sm_crt.py` checks the CRT with the anchor in RAM and on disk. It also checks that the first wrap never comes before the first step where a wrap was possible. With 12-bit counts, for example, a wrap was first possible at step 12 and first happened at step 25.
- At n = 47, per-step checksums match the unmodified code with and without spilling. With a 0.2 GB spill limit, 1,246 runs were spilled and peak memory dropped from 1,069 MB to 959 MB. That run took 41 s, against 66 s for the old code.

The cost: the anchor can hold up to 47 GB of RAM for the rest of the run, and that is taken off every later step's budget. `--crt-snap-gb 0` sends it to disk instead.

On the n = 75 run's remaining steps, I'd expect every step to stay in RAM with no spilling; `top` should show the merge phase near 6000%.

Files are in `~/Programming/Combinatorics/hamiltonian`:
- `hamiltonian/claude/ham_dp_c.c`
- `hamiltonian/claude/ham_dp_c.py`
- `hamiltonian/claude/ham_count.py`
- `tests/test_sm_merge.py`
- `tests/test_sm_crt.py`
- `doc/sm_spill_snapshot_design.md`

Question
========
Look at `data/linux/new_75_sm_new.txt`. It crashed (OOM?) after
step 57. I didn't see any status code.

Answer
======
The crash is almost certainly the OOM killer, but the more important finding is that this n = 75 run would have given a wrong count anyway, and so did the April n = 75 run. I found two counting bugs while investigating; both are now fixed or guarded, and the details are in `doc/frontier16_findings.md`.

**The crash.** The log stops inside step 58, right after `introduce: si=9311M`: a 149 GB input table, with no traceback and no C message. That is what a SIGKILL from the Linux OOM killer looks like. `dmesg -T | grep -iE "killed process|out of memory"` on the machine would confirm it. The run used the binary it compiled at start-up, from before the spill rule.

**Bug 1: back-edges after the 8th were silently dropped.**
- The code kept at most 8 back-edges per vertex and silently dropped the rest. I found it by comparing against an independent Held-Karp count on random graphs.
- It's now fixed: up to 16 back-edges, with a hard error beyond that.
- No logged run reached 8 back-edges (G_n peaks at 5–6), so none of your results are affected.

**Bug 2: an in-step frontier of 16 corrupts the state key.**
- The 64-bit key holds 15 frontier slots plus two flag bits (bits 62–63). The guard let a frontier of 15 through before introducing the next vertex. That vertex then went into a 16th slot that overlaps the flags.
- During a step the frontier is one more than the pw/`max_fw` that `ham_count` prints. So every pw = 15 ordering hits this.
- The evidence is conclusive:
  - With an in-step frontier of at most 15, all 45 random test graphs agree with Held-Karp, as do all 12 perturbed orderings of G_36, G_38 and G_40.
  - At a frontier of exactly 16, 7 of 15 random graphs and all 9 G_n orderings are wrong. For example, G_36 gave 1728, 105 and 61 instead of 12,960.
- n ≤ 74 is unaffected, since those orderings have pw ≤ 14. Both n = 74 runs agree.
- **But n = 75 used pw = 15 orderings in both runs, so the April value 652,211,775,281,872,067,084 should be treated as withdrawn, not just caveated.** That affects `talk2.tex` and the papers.

**What I committed:**
- Orderings that need a 16-slot frontier are now refused up front with a clear message, and the C loops check it too.
- A new `tests/test_dp_correctness.py` checks against Held-Karp. The old code fails all four of its back-edge graphs; the new code passes, alongside the existing merge and CRT tests.

**Proposal, not implemented.** I think the 16th slot fits in the same 64-bit key, with no memory cost, which would make n = 75 possible again:
- The NC flag only matters within the last step. A chain that closes at an earlier step can never be extended into a Hamiltonian path, so that state can be dropped immediately. At the last step the closure is counted directly.
- The MARKER bit only keeps the initial key non-zero. A sentinel value of `UINT64_MAX` can replace it, because that is never a valid key: a label occurs at most twice.

This changes the state semantics, so I'd like you to check that argument before I touch the encoding. Pathwidth 16, which will presumably come at some n > 75, would still need 128-bit keys, at 1.5–2× the memory.

Files are in `~/Programming/Combinatorics/hamiltonian`:
- `hamiltonian/claude/ham_dp_c.c`
- `hamiltonian/claude/ham_dp_c.py`
- `tests/test_dp_correctness.py`
- `doc/frontier16_findings.md`


Question
========
I found that `dmesg` didn't accept the `-T` argument on that machine,
so I couldn't get any useful information. I'm glad that you found the
bug. If this changes the state semantics, the conservative thing to do
is to keep the existing semantics (perhaps in a separate file), so we
can always compare, and revert if necessary. I'd say to go ahead with
the new semantics.

Answer
======
The 16-slot key is in, all the new tests pass here and on the Linux VM, and the original encoding is kept for comparison.

- **Two variants.** `ham_dp_c.c` now has the new key: 16 frontier slots and no flag bits. The old 15-slot key with the two flags is frozen in `ham_dp_c_legacy15.c`. Apart from the encoding, the legacy file has all of yesterday's fixes. You choose with `ham_dp_c.set_c_variant("legacy15")` or `HAM_DP_VARIANT=legacy15`; the frontier-width check follows whichever is selected.
- **What changed in the counting.** If a path closes before the last step, that state is now discarded at once; if it closes at the last step, it is counted at once. Nothing has to remember a "closed" flag between steps any more. The other details are in §5 of `doc/frontier16_findings.md`.
- **Old checkpoints.** The checkpoint format version is bumped, so EH checkpoints written with the old key are rejected rather than misread.

Checks, all exact (paths and cycles, for both the sort-merge and hash-table back ends):

| check | result |
|---|---|
| random graphs needing 14, 15 or 16 slots, against Held-Karp | 51 / 51 exact |
| G_36, G_40, G_44, new key against legacy | identical; G_44 is 5,301,127 paths and 10,091 cycles |
| eight orderings of G_36/40/44 that need 16 slots | all exact; legacy refuses them |
| G_47 against the original code | same count, 68,361,609 |
| merge and CRT tests | pass with both variants |

As designed, an ordering that needs 17 slots (pathwidth 16) is refused with a clear message; it would need a 128-bit key.

**A hiccup during the commit.** The first copy of `ham_dp_c.c` to your Mac silently delivered yesterday's file. The VM test run caught it, I recommitted, and the checksum on your Mac now matches mine. The other files were correct the first time.

**For n = 75.** The pw = 15 orderings now run correctly, so a rerun with the new code gives a trustworthy count. The April value should stay withdrawn until it's reproduced.

Before rerunning, two things:
- **If it gets killed again,** `dmesg | grep -iE "killed process|out of memory"` works without `-T`, or try `journalctl -k`. Either should confirm whether it was the OOM killer.
- **The spill directory.** `findmnt /tmp` and `df -h /tmp` will show whether `/tmp` is a real disk with enough room; the spill rule may write runs there. If it isn't, pass `--spill-dir` pointing at a disk.

Files are in `~/Programming/Combinatorics/hamiltonian`:
- `hamiltonian/claude/ham_dp_c.c`
- `hamiltonian/claude/ham_dp_c_legacy15.c`
- `hamiltonian/claude/ham_dp_c.py`
- `tests/test_dp_correctness.py`
- `doc/frontier16_findings.md`

Question
========
One striking thing. I ran n=58 on my iMac which has an M3, and on an
older laptop which has an M2. The older laptop has 12 cores, and the
iMac 6, but the iMac takes around 122 seconds, and the M2 29. I would
have expected a speedup of about 2x but not 4x. You can see the
`machine.yaml` and `work_model.json` in the main directory. There are
two others I copies from the M2 machine prefixed with `sri` but using
those on the M3 made no difference.

Could the m3 have been reusing the old compiled files? How do you
decide whether to recompile or not? Should there be CLI to delete the
files. I would think that you should key them via a hash of the
relevant inputs. That would be the safest way.

Answer
======
The M3 couldn't have picked up a binary built from an older version of `ham_dp_c.c`. It could, though, be running a binary built by a different compiler than you think, and the compiler choice is the likeliest cause of the 4×. The interrupted VM run isn't still going; I checked, and only my own shells were left.

**How it decides today** (`_get_lib` in `ham_dp_c.py`):
- The cache key is `md5(ham_dp_c.c)[:12]` plus the two compile-time constants (`EH_BKT_CAP`, `RSORT_BITS`).
- The library is stored as `ham_dp.so` in `$TMPDIR/ham_dp_c_<key>/`. On macOS that's a per-user `/var/folders/...` directory, so the two Macs never share builds.
- If that file already exists, it is loaded without recompiling.

So any edit to the C source gives a new key and a fresh build. But the key leaves out several things that affect the binary:
- **Which compiler.** It uses `gcc-15` if that's on `PATH` and plain `gcc` otherwise. On a Mac, plain `gcc` is Apple clang. If the M2 laptop has Homebrew's gcc-15 and the iMac doesn't, the two machines run differently compiled code. Clang and gcc can differ a lot on this kind of code (`__int128` arithmetic, the label loops in `canon`). That fits a ratio beyond the 2× you'd expect from 4 versus 8 P-cores. Nothing is printed to say which compiler was used, so the log can't settle it. `which gcc-15` on each machine would.
- **The compiler's version and the flags.** An upgraded gcc-15 or a change to `-O3 -march=native` would silently reuse the old `.so`.
- **Incomplete or concurrent builds.** If gcc is interrupted it can leave a partial `.so` that later fails to load, and two processes building the same key at once can collide.

On `sri-machine.yaml`: it holds only the probe's informational keys (written by `--probe-only`) and no `SM_*` tuning values. So on the iMac those runs used the iMac's own defaults, which is why it made no difference. Separately, the probe's L1 measurement fails natively on Apple Silicon (k\* = 2 at every stride, reported as null), so that needs a look too.

**What I'd change.** This is your hash-of-all-inputs idea; kobayashi-pw's `native.py` already does it this way.
- **Key:** sha256 over the C source, the compiler's resolved path and `--version` output, all flags and `-D` constants, the C variant (`key16` or `legacy15`), and the platform and machine type.
- **Storage:** a persistent cache directory, `$HAM_DP_CACHE` or `~/.cache/hamiltonian/ham_dp_c/<key>/`. Builds happen under a file lock and are renamed into place only once complete, so a half-built file can never be loaded.
- **Log line:** each run prints `# ham_dp_c: compiled` or `cached`, with the key, compiler, version and flags.
- **Commands:**
  - `python -m hamiltonian.claude.ham_dp_c --list-cache` and `--clear-cache`;
  - `ham_count --rebuild` to force a fresh build;
  - `--cc PATH` (or `HAM_DP_CC`) to choose the compiler explicitly, so you can compare clang and gcc-15 on the same machine.

Shall I implement that? Then an instrumented n = 58 run on each Mac
(`--instrument`, a couple of minutes) would show the compiler in the
log and split the time into compute, sort and merge, so the 4× can be
attributed properly.

Question
========
Yes, implement that.

Answer
======
The build cache is in and committed, with tests passing here and on the Linux VM. To your question: no, it doesn't look at `$CC`.

**How it picks the compiler**, first match wins:
1. `ham_count --cc PATH`, or `set_compiler(path)` from Python;
2. the environment variable `$HAM_DP_CC`;
3. `gcc-15`, if it is on `PATH`;
4. otherwise plain `gcc`. On a Mac without Homebrew gcc, that is Apple clang.

I left `$CC` out because it is often set for other builds, for example by conda or Homebrew, and it would silently change which compiler this uses. Now that the compiler is part of the cache key and printed on every run, honouring it would be safe. If you'd like that, I'd put `$CC` between steps 2 and 3.

**How the cache works now:**
- **Key:** a SHA-256 over the C source and variant (`key16` or `legacy15`), the compiler's resolved path and full `--version` output, every flag and `-D` constant, and the OS and CPU type. Changing any of these gives a new build.
- **Storage:** `$HAM_DP_CACHE`, defaulting to `~/.cache/hamiltonian/ham_dp_c/<key>/`, with a `meta.json` recording those inputs. It no longer uses `$TMPDIR`.
- **Builds:** done under a file lock and moved into place only when complete, so a half-built library can never be loaded, and a forced rebuild happens only once per run.
- **Log:** every run prints whether the library was compiled or cached, the key, the compiler and version, and the flags. It also prints the settings in effect: `SM_NTHREADS`, `SM_WORKER_CAP`, `SM_RBUF_SIZE`, `SM_BB_BUF`, `SM_PAR_MERGE_THRESH`.
- **Commands:**
  - `python -m hamiltonian.claude.ham_dp_c --list-cache`
  - `python -m hamiltonian.claude.ham_dp_c --clear-cache`, which also removes old-style `$TMPDIR/ham_dp_c_*` builds
  - `python -m hamiltonian.claude.ham_dp_c --build [--cc …] [--variant …]`
  - `ham_count --rebuild`, `--cc PATH` and `--dp-variant {key16,legacy15}`

**For the M3 against M2 question:** run n = 58 once on each machine with `--instrument`. The header lines will show which compiler each machine really used, and the thread count. To compare compilers on the iMac, add `--cc gcc-15` (if it's installed) or `--cc clang`. Each compiler gets its own cached build, so you can switch back and forth without rebuilding.

Files are in `~/Programming/Combinatorics/hamiltonian`:
- `hamiltonian/claude/ham_dp_c.py`
- `hamiltonian/claude/ham_count.py`
- `tests/test_build_cache.py`

Question
========
There are two C++ programs in `/Users/victorsmiller/Programming/Combinatorics/hamiltonian/doc/contrib` which calculate the number of Hamiltonian paths for the square graph written by others. They both use the Held-Karp DP but it's not clear what vertex ordering they use. Can you look at them and write a markdown doc which analyzes them, and points out possible shortcomings?

Answer
======
I've written `doc/contrib_analysis.md`. Neither program actually uses Held-Karp, and `A071984_opt.cpp` counts Hamiltonian cycles, not paths. I compiled both and ran them in the Linux VM with 4 threads; the timings below are single runs.

**`A071984_opt.cpp` (Dobbelaere)** is a plain backtracking search with no memoization.
- It counts each cycle once by forcing it to start through a vertex that has only two neighbours (for example 14–50–31 when 50 ≤ N ≤ 70). I checked the degrees and the trick is sound.
- Its only pruning is a degree check on the vertices left behind.
- There is no real vertex ordering, just the order the search happens to extend the path.
- Limits: it stops at N = 70 and runs on one thread. Turning it into a path counter would break the start-vertex trick and most of the pruning.
- It reproduced its own commented values for N = 32–52 in about 2 minutes. The time grows about 2.3× per step in N.

**`path4.cpp` (no author named)** is a frontier DP whose state is the same kind as ours: the endpoint pairs of path fragments, plus a bitmask of vertices that already have two edges.
- **Ordering:** fixed, N, N−1, …, 1. Its frontier peaks at about 0.7·N, which is 40 at N = 58. Our ordering gives pathwidth 12 there.
- **Endpoints:** it adds a pseudo-edge between the two path endpoints and starts with all N(N−1)/2 endpoint pairs as separate states. That multiplies the state count, because pairs of small vertices survive almost to the end.
- **The DP is only used at the top:** states are merged for the first 7 vertices only. After that it is an unmemoized search. I tried running the merged DP all the way down, and that was slower (N = 42 didn't finish in 40 s, against 11.5 s for the original).
- **Speed:** it matches `data/b071983.txt` for N ≤ 44. N = 42 took 11.5 s and N = 44 took 131 s, so N = 58 is out of reach.
- **Other costs and limits:** it stores states in a `std::map` and rebuilds a union-find for every new state. It only works for N ≤ 62 with 64-bit `long`, and fails for N ≥ 32 on Windows, where `long` is 32 bits. One of its pruning tests is weaker than it could be.
- **Correctness:** I found no bug, but I haven't proved its pair-list update routines correct line by line; agreement with the reference data is evidence, not proof.

`path4` has one idea we don't use: pruning states whose remaining graph is disconnected, and forcing an edge when an endpoint has only one option left. With our much smaller frontier it's unclear whether that would pay off. It could be measured cheaply by counting how many of our new states would fail the connectivity check.

Question
========
The author of `path4.cpp` is Zhao Hui Du, a frequent contribution of
OEIS. It's worthwhile to test his idea about disconnected graphs.

Answer
======
I'll write a short design note first, then add the check to the key16 code behind an environment variable that is off by default, and measure it.

Pruned and unpruned counts agree with `b071983.txt` for every n from 15 to 50. Next, timing at n = 58:

While that runs, I'll put Zhao Hui Du's name into the contrib analysis.

Pruning gives n = 58 in 130 s in the VM; without it the run had only reached step 37 after 158 s. Now the test suite, with pruning on and off:

Connectivity never removed a state that the endpoint budget had kept; the budget alone accounts for all of the 55% at n = 52. I'll write the results into the design note and commit.

Du's idea pays off, but not through connectivity. A simpler test that comes with it, an "endpoint budget", does all the work. With it on, the counts are unchanged and n = 57 and 58 run about 2.3× faster in the Linux VM.

**What I added.** The check is in the key16 C file, turned on by `HAM_SM_PRUNE`; it is off by default. `HAM_SM_PRUNE=1` only reports how many states are dead, `HAM_SM_PRUNE=2` drops them. I wrote the design note before the code; it's `doc/sm_prune_design.md`, with the results added at the end. After each step, two necessary conditions for a state to be completable are checked:
- **(D) Endpoint budget:** a Hamiltonian path has only two ends. Some vertices are already forced to be ends: the far end of a chain that has left the frontier, an untouched frontier vertex with only one later neighbour, and a later vertex with only one usable neighbour. If more than two are forced, the state is dead.
- **(C) Connectivity:** the unfinished frontier vertices, the vertices not yet processed, and the existing chains must form one connected graph. This is Du's test.

Our DP had neither check. A dead state, such as one with three dangling chains, was carried and multiplied to the very last step before it died.

**Correctness (VM, 4 threads):**
- **Counts:** with pruning on, path counts are unchanged and match `b071983.txt` for every n from 15 to 50, and for 53 and 55–58. Cycle counts match for n = 32–48, including Dobbelaere's values.
- **Tests:** the full correctness suite passes, including the random width-16 graphs checked against Held-Karp. Two other tests fail with pruning on: one expects a 12-bit overflow and the other expects spills, and pruning makes the counts and tables too small for either to happen. Both pass with pruning off.

**What it finds.** About 55% of all states at n = 52 are dead, and 25–75% per step at n = 40. Every one of them fails the endpoint budget; connectivity never removes a state the budget would have kept.

**Speed:**

| n | Unpruned | Pruned |
|---|---|---|
| 55 | 9.4 s | 5.2 s |
| 56 | 16.6 s | 11.2 s |
| 57 | 68.2 s | 29.4 s |
| 58 | not finished: step 37 after 158 s | 130 s total (step 37 at 65 s) |

At n = 58, the table at step 37 shrinks from 24.5 M states to 7.5 M. These are single runs each, and the VM's orderings differ from the Mac's.

**Still to do:**
- find out which part of the endpoint budget fires;
- probably drop the connectivity test, which costs time and removes nothing extra;
- prune before generating each state's successors, not only after;
- run it at n = 69 or 74 on your machines, e.g. `HAM_SM_PRUNE=2 ham_count 69 …`, before making it the default.

I also added Zhao Hui Du's name to `doc/contrib_analysis.md`.

My `git commit` failed: the VM has no git identity, and deleting files
in the folder isn't allowed. The three files are staged in your index
but not committed. That step also left a couple of stray temporary
files under `.git/objects`, which are harmless.

Question
========
Look at `count_58_sm_pruned2.txt` and `count_58_sm_newest.txt` in the
main directory. The first uses HAM_SM_PRUNE=1 and the latter
doesn't. On the positive side there are a lot of pruned states, but
it's 40% slower. This is rather puzzling.

Answer
======
The slowdown is expected: mode 1 doesn't prune anything. That's my naming at fault. `HAM_SM_PRUNE=1` is report-only: it keeps every state, so the DP does the same work as an unpruned run. On top of that, after each step's merge it classifies the whole merged table on one thread, purely to print the `[prune] … deadD=… live=…` line. At steps 30–42 that means scanning 10–15 M states per step serially, which adds up to the extra time you saw. The mode that drops dead states is `HAM_SM_PRUNE=2`.

There is a second, smaller confound: the two runs used different orderings, so even mode 2 against `count_58_sm_newest.txt` wouldn't be a clean comparison.

| | `count_58_sm_newest.txt` | `count_58_sm_pruned2.txt` |
|---|---|---|
| machine yaml | `sri-machine.yaml` | `machine.yaml` |
| kpw ordering | labels=318, value 4.974e9 | labels=200, value 7.818e8 |

The orderings differ in places, for example at step 32, where the newest run has 12.46 M output states and the pruned2 run 10.12 M.

For a clean A/B, run both with the same yaml:
```
HAM_SM_PRUNE=2 ham_count 58 … --machine-yaml machine.yaml
HAM_SM_PRUNE=0 ham_count 58 … --machine-yaml machine.yaml
```

Your run does confirm the potential. At steps 40–42, between 36% and 47% of the states are live. With mode 2 those dead states are never written, so their successors at the next step are never generated either.

To avoid the same confusion later, I'd like to replace the numbers
with names, `HAM_SM_PRUNE=report` and `HAM_SM_PRUNE=drop`, and make
the report pass multithreaded. Should I make that change?

Question
========
I just reran it with HAM_SM_PRUNE=2. Now it takes 49 seconds - a
little over 2x. I usually don't like passing parameters via the
environment. Could you make it a command line option - make drop the
defulat, report and none as options. Also, can you make kpw with the
dsl the default now instead of maxsat? Also, I've made kobayashi-pw
public on github, so you should adjust pyproject.toml.

