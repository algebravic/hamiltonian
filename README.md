The OEIS Sequence A090460
=========================

The sequence A090460 gives the number of essentially different
permutations of the numbers 1 to $n$ such that the sum of adjacent
numbers is a square. There is no indication there of how the larger
values are calculated. There is a mathematica program given which
calculates them by brute force enumeration, using a simple backtrack
algorithm.

If $G$ is a connected undirected graph, a *Hamiltonian Path* is path
consisting of edges which visit every vertex. More specifically, if
$G = (V,E)$, then a Hamiltonian path is a sequence of distinct vertices
`$v_1, \dots, v_n$`, where `$n=\#V$`, such that `$(v_i, v_{i+1}) \in E$`
for $i=1,\dots, n-1$.  Given $n$, construct the graph $G$ whose
vertices are labeled $1, \dots, n$, and where $(i,j) \in E$ if and
only if $|i-j|$ is a positive square. Then A090460 asks for the number
of Hamiltonian paths in $G$.

The graph package ~Graphillion~ using ZDD's (Zero suppressed binary
decision diagrams) to represent the set of all Hamiltonian paths in a
graph. It then can easily (linear in the size of the ZDD) calculated
the number of such paths.

Posa rotation
==========================

If $G=(V,E)$ is a connected undirected graph of cardinality $n$ and
`$(v_1, \dots, v_n)$` is a Hamiltonian path, if there is are edges
`$(v_n, v_i) \in E$` for $j \ne i$, then
`$(v_1, \dots, v_i, v_n, v_{n-1}, \dots, v_{i+1})$` is also a
Hamiltonian path.

Question: Is it possible to compute the number of equivalence classes
under Posa rotation?

More generally: A set of edges of $G$ is *full* if every vertex in $G$
is incident to at least one edge, the set has cardinality $n-1$, and
$n-2$ of the vertices in $G$ is incident to at exactly 2 edges, and
the remaining 2 vertices only incident to 1 edge. A Posa rotation of
a full set consists in taking one of the end edges (the two edges
incident to the 2 end vertices), and replacing it with another edge
incident to the end.
