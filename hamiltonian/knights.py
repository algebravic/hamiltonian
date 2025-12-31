"""
  Knight's tour graph
"""
from itertools import product
import networkx as nx
from .util import GraphOrder, get_count

def knight_graph(num: int, mnum: int | None = None) -> nx.Graph:

    pnum = num if mnum is None else mnum
    nbrs = {(1,2), (1,-2), (-1, 2), (-1, -2),
            (2, 1), (2,-1), (-2, 1), (-2, -1)}
    dims = (num, pnum)
    board = set(product(range(num), range(pnum)))
    gph = nx.Graph(name = f'knight{dims}')
    for square in board:
        for delta in nbrs:
            gph.add_edge(square, (square[0] + delta[0], square[1] + delta[1]))
    gph.remove_nodes_from([_ for _ in gph if _ not in board])
    return gph
