"""
  Number of hamiltonian paths of the hamming cube.
"""

from typing import List
from math import floor, sqrt, ceil
from itertools import product
import networkx as nx
from graphillion import GraphSet
from .util import use_networkx

def hamming_graph(num: int) -> nx.Graph:
    """
      The Hamming cube of dimension n
    """
    gph = nx.Graph()
    for pnt in product(range(2), repeat=num):

        gph.add_edges_from((pnt, pnt[: _] + (1 - pnt[_],) + pnt[_ + 1: ])
                           for _ in range(num))
    return gph

def get_hamming_count(num: int):

    gph = hamming_graph(num)
    return GraphSet.paths(None, None, is_hamilton=True).len()
