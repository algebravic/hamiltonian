
"""Use the MILP formulation given in
  https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/graph_decompositions/vertex_separation.html

  Note: According to the paper of Kinnersley:

  V_L(i) is the number of vertices of G mapped to integers <= i that are adjacent
  to integer greater than i.

  In the model below this is reversed: V_L(i) is the number of vertices of G that
  mapped to integers > i that are adjacent to vertices <= i.  This seems to be the reverse

  This model is due to David Coudert
  
  rendered as a MaxSat Problem using RC2.

  We describe below a mixed integer linear program (MILP) for
  determining an optimal layout for the vertex separation of G, which
  is an improved version of the formulation proposed in [SP2010]. It
  aims at building a sequence S[t] of sets such that an ordering of
  the vertices correspond to S[1] = {v[1]}, S[2] = {v[1], v[2]}, ...,
  S[n] = {v[1], ..., v[n]}.

  My comment: Let U[t] denote the subset of nodes which are not in S[t]
  but have a neighbor in S[t]

  X[t] = S[t] union U[t]

  Requirement #S[t] = t
  S[t] subset S[t+1]
  X[t] = neighbors(S[t])
  so X[t+1] = X[t] union {v[t]} union neighbor({v[t]})

  does U[t] = X[t] difference S[t]?

  i.e. u[v,t] = ~y[v,t] AND x[v,t]
  or y[v,t] OR ~x[v,t] OR u[v,t] is true.
  and ~u[v,t] OR ~y[v,t], ~u[v,t] OR x[v,t]

  But, according to Kinnersley we should have
     U[t] = { v in S[t] AND exists w not in S[t] for w in neighbors(v)}
  So u[v,t] = y[v,t] AND (OR (~y[w,t] for w in neighbors(v)))
  Since u is bounded from above we only need one direction:

  ~y[v,t] OR (~ (OR (~y[w,t] for w in neighbors(v)))) OR u[v,t]


  ~y[v,t] OR u[v,t] Or y[w,t] for all w in neighbors(v)

  If w is not in S[t] and v is in S[t] then u[v,t] is true
  If w is in S[t] or v is not in S[t] then no condition on u[v,t]
  
  In the other direction:
  ~u[v,t] OR (y[v,t] AND (OR (~y[w,t] w in nbr(v)))) <==>
  ~u[v,t] OR y[v,t]
  ~u[v,t] OR OR(~y[w,t] for w in nbr(v))
  
  So u[v,t] = y[v,t] AND ~x[v,t]
  So we have

  ~y[v,t] OR x[v,t] OR u[v,t]
  ~u[v,t] OR y[v,t]
  ~u[v,t] OR ~x[v,t]
  
  Variables:

  . y[v,t] – variable set to 1 if v in S[t], and 0 otherwise. The order
  of in the layout is the smallest t such that y[v,t] = 1

  . u[v,t] – variable set to 1 if v not in S[t] and has an in-neighbor
  in S[t]. It is set to 0 otherwise.

  . x[v,t] – variable set to 1 if either v in S[t] or if v has an
  in-neighbor in S[t]. It is set to 0 otherwise.

  . z – objective value to minimize. It is equal to the maximum over
  all step t of the number of vertices such that y[v,t] = 1.

  MILP formulation:

  Minimize: z
  Such that:
  x[v,t] <= x[v,t+1] for v in V, 0 <= t <= n-2
  y[v,t] <= y[v,t+1] for v in V, 0 <= t <= n-2
  y[v,t] <= x[w,t] for v in V, w in N+(v), 0 <= t <=n-1
  sum(v in V) y[v,t] = t+1 for 0<= t <= n-1
  x[v,t] - y[v,t] <= u[v,t] for v in V, 0<=t <= n-1
  sum(v in V) u[v,t] <= z for 0<=t <= n-1
  0 <= x[v,t] <=1 for v in V, 0 <=t <= n-1
  0 <= u[v,t] <=1 for v in V, 0 <=t <= n-1
  y[v,t] in {0,1}
  0 <= z <= n


  The vertex separation of G is given by the value of z, and the order
  of vertex v in the optimal layout is given by the smallest t for
  which y[v,t] = 1.

  Note that this is for directed graphs, N+(v) is the out neighborhood of v

"""
from typing import Dict, Hashable, Tuple, List
from itertools import product, chain
import networkx as nx
from pysat.formula import IDPool, WCNF
from pysat.examples.rc2 import RC2, RC2Stratified
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver

class VertexSeparation:

    def __init__(self, gph: nx.Graph, encode='totalizer'):

        self._graph = gph
        self._pool = IDPool()
        self._cnf = WCNF()
        self._size = len(self._graph.nodes)
        self._encode = getattr(EncType, encode, EncType.totalizer)
        self._model()
        
    def _model(self):

        # Hard constraints
        # #   x[v,t] <= x[v,t+1] for v in V, 1 <= t <= n-1
        # self._cnf.extend([
        #     [-self._pool.id(('x', _)), self._pool.id(('x', _[0], _[1] + 1))]
        #     for _ in product(self._graph.nodes, range(1, self._size))])
        #   y[v,t] <= x[v,t] for v in V, 1 <= t <= n-1
        # self._cnf.extend([
        #     [-self._pool.id(('y', _)), self._pool.id(('x', _))]
        #     for _ in product(self._graph.nodes, range(1, self._size + 1))])
        #   y[v,t] <= y[v,t+1] for v in V, 1 <= t <= n-1
        self._cnf.extend([
            [-self._pool.id(('y', _)), self._pool.id(('y', (_[0], _[1] + 1)))]
            for _ in product(self._graph.nodes, range(1, self._size))])
        # #   y[v,t] <= x[w,t] for v in V, w in N+(v), 1 <= t <=n
        # self._cnf.extend([
        #     [-self._pool.id(('y', _)), self._pool.id(('x', (nbr, _[1])))]
        #     for _ in product(self._graph.nodes, range(1, self._size + 1))
        #     for nbr in nx.neighbors(self._graph, _[0])])
        # z non-increasing
        self._cnf.extend([[self._pool.id(('z', _)), -self._pool.id(('z', _ + 1))]
                          for _ in range(1, self._size)])
        zneg = [-self._pool.id(('z', _)) for _ in range(1, self._size + 1)]
        for tme in range(1, self._size + 1):
            self._cnf.extend([
                [-self._pool.id(('y', (node, tme))),
                 self._pool.id(('u', (node, tme))),
                 self._pool.id(('y', (nbr, tme)))]
                for node in self._graph.nodes
                for nbr in nx.neighbors(self._graph, node)])

            # self._cnf.extend([
            #     [-self._pool.id(('y', (node, tme))),
            #      self._pool.id(('u', (node, tme))),
            #      self._pool.id(('y', (nbr, tme)))]
            #     for node in self._graph.nodes
            #     for nbr in nx.neighbors(self._graph, node)])

            # Objective to be minimized
            self._cnf.append([-self._pool.id(('z', tme))], weight=1)
            #   sum(v in V) y[v,t] = t for 1 <= t <= n
            ylits = [self._pool.id(('y', (_, tme))) for _ in self._graph.nodes]
            self._cnf.extend(CardEnc.equals(lits = ylits,
                                            bound = tme,
                                            encoding = self._encode,
                                            vpool = self._pool))
            ulits = [self._pool.id(('u', (_, tme))) for _ in self._graph.nodes]
            #   sum(v in V) u[v,t] <= z for 1 <= t <= n
            self._cnf.extend(CardEnc.atmost(lits = ulits + zneg,
                                            bound = self._size,
                                            encoding = self._encode,
                                            vpool = self._pool))
        #   x[v,t] - y[v,t] <= u[v,t] for v in V, 1 <= t <= n
        # self._cnf.extend([
        #     [self._pool.id(('y', _)), self._pool.id(('u', _)), - self._pool.id(('x', _))]
        #     for _ in product(self._graph.nodes, range(1, self._size + 1))])
        #   x[v,t] - y[v,t] <= u[v,t] for v in V, 1 <= t <= n
        # self._cnf.extend([
        #     [- self._pool.id(('y', _)), self._pool.id(('u', _)), self._pool.id(('x', _))]
        #     for _ in product(self._graph.nodes, range(1, self._size + 1))])


    def solve(self,
              solver = 'cd195',
              stratified: bool = False,
              **kwds) -> Tuple[int, Dict[Hashable, int]]:

        maxsat_solver = RC2Stratified if stratified else RC2
        max_solver = maxsat_solver(self._cnf, solver = solver, **kwds)
        soln = max_solver.compute()
        if kwds.get('verbose', 0) > 0:
            print(f"Time = {max_solver.oracle_time()}")
        pos = [self._pool.obj(_) for _ in soln if _ > 0]
        zvals = [_[1] for _ in pos if _ is not None and _[0] == 'z']
        yvals = {_[1] for _ in pos if _ is not None and _[0] == 'y'}

        yorder = {node: min((_[1] for _ in yvals if _[0] == node))
            for node in self._graph.nodes}
        return len(zvals), yorder
        
def pathwidth_order(gph: nx.Graph, **kwds) -> Tuple[int, List[Hashable]]:

    """
      Produce a renumbered graph by means of optimal pathwidth (the
      same as vertex separation order).
    """

    vsp = VertexSeparation(gph)
    sep, renumber = vsp.solve(**kwds)
    return sep, [_[0] for _ in sorted(renumber.items(), key=lambda _: _[1])]

def separation(gph: nx.Graph) -> int:

    snodes = sorted(gph.nodes)

    val = 0
    # Calculate V_L(i)
    for ind in range(len(gph.nodes)):
        tnodes = set(snodes[: ind + 1])
        bad = {node for node in tnodes
            if not set(nx.neighbors(gph, node)).issubset(tnodes)}
        val = max(len(bad), val)
    return val

def alt_separation(gph: nx.Graph) -> int:

    snodes = sorted(gph.nodes)

    val = 0
    # Calculate V_L(i)
    for ind in range(len(gph.nodes)):
        tnodes = set(snodes[: ind + 1])
        bad = set(chain(*(set(nx.neighbors(gph,node)).difference(tnodes)
                        for node in tnodes)))
        val = max(len(bad), val)
    return val
