"""
  Add SAT constraints which indicate that an edge induced subgraph
  forms a Hamiltonian cycle.

  We are given an undirected networkx graph. If the order of the graph is n
  we seek a set of n-edges which forms an n-cycle.

  Strategy: We have variables which indicate whether or not the edges are
  in the cycle, and another set of variables which indicate the direction
  (ignored if an edge is not in the cycle).
  Constraints:
  1) Every vertex has precisely one in-edge and one out-edge.
  2) Connectedness: We choose one vertex, arbitrarily (is there a better choice?).
  Attached to every vertex is a length variable l[v].

  We should have, for all vertices w not the initial vertex, v[0]

  l[w] = 1 + l[v], where (v,w) is the unique in-edge to w.
  l[v[0]] = 0

  Since v[0] has a unique in-edge, and w so that l[w] = n-1 has a unique out-edge
  the only place that the out-edge can go from w is to v[0] (check this).

  We can implement the l[v] variables either using log n bits of using n bits
  (this probably propagates better).

  Details: Let e[v,w] be a boolean variable that, if true, says that (v,w)
  is in the cycle. Let d[v,w] = 0 if the direction is v-->w and 1 if w-->v

  for every v in G, we have exactly one of e[v,w] AND d[v,w] (in edges) is true
  and exactly one of e[v,w] AND ~d[v,w] is true (out-edges).

  We have variables L[v,i] for i=0,...,n-1, where L[v,i] = True if and only if
  l[v] = i. So we have exactly one_i (L[v,i]) for all v.

  We have, for all v, and all (v,w) in  (Or_v L[v,i] AND e[v,w] AND d[v,w]) ==> L[w,i+1]
  That is, if w has an incoming edge (v,w) and l[v] = i then l[w] = i+1

  We also need the converse

  This is the same as (AND_w (~L[w,i] OR ~e[v,w] OR ~d[v,w]) OR L[v,i+1]

  Or_v M[v,w,i] <== L[w, i + 1]

  Perhaps there are better variables:

  i[v,w] : the edge v--> w is present
  o[v,w] : the edge w-->v is present

  Both can't be true.
  
  Then i[v,w] OR o[v,w] <==> e[v,w] (Note that we no longer need e[v,w])
  I think that we don't need o[v,w] either. Maybe only for v = v[0]. No, all of
  the other nodes will have precisely one in neighbor, so the only one left is
  v[0].

  M[v, w,i] <===> i[v,w] AND L[v,i].

  That is, (v,w) is an incoming edge of w and l[v] = i

  In particular M[v[0], w, 0] = i[v[0], w], since l[v[0]] = 0

  We then have (AND_v ~M[v, w ,i]) OR L[w,i+1]

  Note that every node != v[0] will have l[w] > 0. If w is a neighbor of v[0]
  Then M[w,1] <==> i[v[0], w]

  At the end, some node, w should have L[w,n-1] == True

  This roughly should have |V|^2 |E| clauses and |V|^2 + |E| variables.
  
"""
from typing import List, Iterable
import networkx as nx
from pysat.formula import IDPool
from pysat.card import CardEnc, EncType

CLAUSE = List[int]

def hcycle(gph: nx.Graph, pool: IDPool,
           optional: bool = False,
           encode: str = 'totalizer') -> Iterable[CLAUSE]:

    # Find v[0].
    # Each edge can only be oriented in one way

    num = gph.order()

    # Find a vertex of minmum degree:

    mindeg = min((gph.degree(_) for _ in gph.nodes))
    min_nodes = [_ for _ in gph.nodes if gph.degree(_) == mindeg]
    print(f"There are {len(min_nodes)} of degree {mindeg}")
    root = min_nodes[0]

    encoding = getattr(EncType, encode, EncType.totalizer)

    for node in gph.nodes:
        # L variables
        # Every node has a length
        lvars = [pool.id(('L', node, _)) for _ in range(num)]
        yield from CardEnc.equals(lits = lvars,
                                  bound = 1,
                                  encoding = EncType.ladder,
                                  vpool = pool)

    if optional:
        # length has a unique node
        # This isn't strictly necessary, but will help    
        for lens in range(num):
            ltvars = [pool.id(('L', _, lens)) for _ in gph.nodes]
            yield from CardEnc.equals(lits = lvars,
                                      bound = 1,
                                      encoding = EncType.ladder,
                                      vpool = pool)
        # This also isn't strictly necessary but will also help
        # There are exactly n incoming edges
        oriented = ([pool.id(('i',) + _) for _ in gph.edges] +
                    [pool.id(('i', _[1], _[0])) for _ in gph.edges])
        yield from CardEnc.equals(lits = oriented,
                                  bound = num,
                                  encoding = encoding,
                                  vpool = pool)

        # Some Node has l[v] = n-1
        yield [pool.id(('L', _, num - 1)) for _ in gph.nodes]
        
    yield [pool.id(('L', root, 0))] # Root gets length 0

    # An edge can only be oriented in one way
    yield from ([-pool.oid(('i', inv, outv)), -pool.id(('i', outv, inv))]
                for inv, outv in gph.edges)

    for node in gph.nodes:
        # Exactly one outgoing edge
        yield from CardEnc.equals(lits = [pool.id(('i', node, _))
                                          for _ in gph.neighbors(node)],
                                  bound = 1,
                                  encoding = EncType.ladder,
                                  vpool = pool)
        # Exactly one incoming edge
        yield from CardEnc.equals(lits = [pool.id(('i', _, node))
                                          for _ in gph.neighbors(node)],
                                  bound = 1,
                                  encoding = EncType.ladder,
                                  vpool = pool)

        # incoming edges
        ivars = [pool.id(('i', nbr, node)) for nbr in gph.neighbors(node)]
        for ind in range(num):
            # Possible in-nodes
            # i[w,v] AND L[w,i] ==> L[v, (i+1) % n]
            lpvar = pool.id(('L', node, (ind + 1) % num))
            lvars = [pool.id(('L', nbr, ind)) for nbr in gph.neighbors(node)]
            yield from ([-ivar, -lvar, lpvar] for lvar, ivar in zip(lvars, ivars))
