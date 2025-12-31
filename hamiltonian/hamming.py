"""
  Number of hamiltonian paths of the hamming cube.
"""

from itertools import product
import networkx as nx

def hamming_graph(num: int) -> nx.Graph:
    """
      The Hamming cube of dimension n
    """
    gph = nx.Graph(name=f'hamming({num})')
    for pnt in product(range(2), repeat=num):

        gph.add_edges_from((pnt, pnt[: _] + (1 - pnt[_],) + pnt[_ + 1: ])
                           for _ in range(num))
    return gph
