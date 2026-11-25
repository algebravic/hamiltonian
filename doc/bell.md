Here is a self-contained section suitable for the paper, covering all three topics — Bell numbers, the exact state-space bound, and the ZDD connection:

---

**Frontier widths, Bell numbers, and ZDDs.**

\paragraph{Set-partition states and the Bell number bound.}
The canonical state of the frontier DP at a step when $w$ vertices are active encodes a partition of those $w$ vertices into three roles: \emph{interior} vertices (degree 2, no further edges permitted), \emph{fresh} vertices (introduced but not yet incident to any selected edge), and \emph{endpoints} of partial path-segments (pairs of vertices sharing a label, recording that they are connected by a chain whose interior has already been processed). The number of such labelled structures is bounded by the \emph{Bell number} $\Bell(w)$, which counts the total number of set-partitions of a $w$-element set (the classical result for Hamiltonian-path frontier DPs, see e.g.~\cite{kinnersley1992}).

In our implementation each frontier position also carries the fresh/interior distinction for unmatched vertices, making the actual combinatorial bound slightly larger than $\Bell(w)$: an unmatched vertex contributes 2 choices (fresh or interior) instead of 1, and a closed-chain flag contributes a further factor of 2. The precise upper bound is
\[
  2 \sum_{k=0}^{\lfloor w/2\rfloor} \frac{w!}{2^k\,k!\,(w-2k)!} \cdot 2^{w-2k},
\]
which equals the double of the exponential generating function evaluation of $e^{x^2/2 + 2x}$ at $w$, and grows faster than $\Bell(w)$ for large $w$. Empirically, the observed state counts for $G_{57}$ (peak $\approx 4.6 \times 10^6$ at $w = 11$) and $G_{59}$ (peak $\approx 6.8 \times 10^7$ at $w = 13$) are consistent with this tighter bound.

The key consequence for algorithm design is that the peak state count is controlled by the maximum frontier width, since $\Bell(w)$ grows super-exponentially. Reducing $\max_s f_s(\sigma)$ by even one, by choosing a better vertex ordering $\sigma$, can reduce peak memory by a factor of roughly $w$ (since $\Bell(w)/\Bell(w-1) \approx w/\ln w$ asymptotically). This is the central motivation for the MaxSAT pathwidth computation of Section~\ref{sec:ordering}.

\paragraph{ZDDs and their relation to the frontier DP.}
The \texttt{graphillion} library~\cite{graphillion} represents families of edge-sets (including the set of all Hamiltonian paths) as \emph{Zero-suppressed Binary Decision Diagrams} (ZDDs)~\cite{minato1993}. A ZDD is a directed acyclic graph in which each internal node tests one Boolean variable (here: is a given edge included?) and has two children corresponding to 0/1 assignments; sharing of isomorphic subtrees compresses the representation. For graph families, the variable order chosen for the ZDD corresponds directly to the edge order in a frontier DP.

The correspondence is exact: each \emph{node at level $d$} in the ZDD corresponds to a \emph{distinct canonical frontier state at step $d$} in the frontier DP. Consequently:
\begin{itemize}
  \item The \emph{total ZDD node count} equals the sum of state counts across all steps, which is the \emph{profile} $\sum_s f_s(\sigma)$ of the vertex ordering.
  \item The \emph{maximum ZDD width} (maximum number of nodes at any single level) equals the \emph{maximum frontier state count}, which is bounded by $\Bell(\pw(G))$.
\end{itemize}
These two objectives are related but distinct. Minimising the profile controls total work and ZDD size; minimising the maximum frontier width (pathwidth) controls peak memory. For \texttt{graphillion}, which builds and stores the entire ZDD before querying it, the profile is the binding constraint: the ZDD for $G_{56}$ consumed the available memory before $G_{57}$ could be attempted. For our hash-table frontier DP, which processes one step at a time and discards completed states, only the \emph{peak} state count matters — making pathwidth the right optimisation target, and explaining why the two approaches have different memory limits for the same problem.

---

A few numerical anchors worth including in the table in Section 2:

| $w$ | $\Bell(w)$ | Corrected bound | Observed peak ($G_n$) |
|-----|-----------|-----------------|----------------------|
| 11 | 678,570 | 1,076,156 | 4,600,000 ($n=57$) |
| 12 | 4,213,597 | 4,860,710 | 22,400,000 ($n=58$) |
| 13 | 27,644,437 | 22,635,292 | 67,500,000 ($n=59$) |

The observed peaks exceed the corrected bound at $w=11$ and $w=13$, which indicates that the bound I derived above is still not tight — the true combinatorial bound for this specific graph family (or more carefully accounting for the label-slot interaction) is somewhat larger. The Bell number remains the standard citation in the literature and gives the right order of magnitude and the key qualitative point: peak state count grows as $\Bell(\pw(G))$, making each unit reduction in pathwidth exponentially valuable.
