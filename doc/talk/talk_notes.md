# Speaker notes: "Counting square chains" (about 50 minutes)

| Part | Slides | Minutes | Point to land |
|---|---|---|---|
| Intro | 1–3 | 4 | Exact counts exist for n ≤ 75. Three questions: is the growth exponential, what is the true rate, and how could one prove it. |
| I. Lifting | 4–9 | 12 | Scaling by 25 preserves square sums. Counting paths loses half the classes (exponent 0.8). Slide 5 shows the gluing on concrete numbers. Parity-preserving moves are multiplied by exactly 25 per level. |
| II. Limits | 10–13 | 12 | Graphon as the limit of a pixel picture. G(n) after rescaling looks like κ = 1/(2√(x+y)), but the residue classes mod 4 show the arithmetic does not average out. The limit has two layers: size and residues. |
| III. Counting | 14–19 | 12 | First moment = energy − entropy over couplings (a Schrödinger bridge); this gives C. The arithmetic washes out at exponential scale. Data fit. The Bregman upper bound B is only 0.0175 above C. |
| IV. Route | 20–23 | 10 | van der Waerden gives exactly C (Lemma L1). Few cycles (L2). Rotations (L3), where the arithmetic bites: the residue quotients are connected even though the graph is not pseudo-random. |

Audience notes:
- Before slide 10, ask who has seen graphons. Slide 10 is self-contained; the names of the frameworks on slide 13 are for orientation only.
- On slide 15 (energy versus entropy), the intuition is: "Hamiltonian cycles would like to use pairs of small numbers, where κ is large, but each vertex must be used exactly once. Entropy prices the departure from a uniform matching."
- On slide 19, stress that B is a theorem, so the remaining question is entirely the lower bound.
- The [proved] / [conjecture] / [heuristic] tags are on the slides on purpose; mention them once at the start.

Regenerate the figures with `uv run python -m hamiltonian.bounds.talk_figures` from the repo root.
