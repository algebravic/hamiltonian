"""
  Grid graph
"""
from itertools import product
import networkx as nx
from .util import GraphOrder, get_count

def grid_graph(num: int, mnum: int = 0) -> nx.Graph:
    mnum = num if mnum == 0 else mnum
    board = set(product(range(num), range(mnum)))
    nbrs = {(1,0), (-1,0), (0,1), (0,-1)}
    gph = nx.Graph()
    for square in board:
        for delta in nbrs:
            nbr = (square[0] + delta[0], square[1] + delta[1])
            if nbr in board:
                gph.add_edge(square, nbr)
    return gph
