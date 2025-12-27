"""
  Knight's tour graph
"""
from itertools import product
import networkx as nx
from .util import GraphOrder, get_count

def knight_graph(num: int) -> nx.Graph:

    nbrs = {(1,2), (1,-2), (-1, 2), (-1, -2),
            (2, 1), (2,-1), (-2, 1), (-2, -1)}
    board = set(product(range(num), repeat = 2))
    gph = nx.Graph()
    for square in board:
        for delta in nbrs:
            nbr = (square[0] + delta[0], square[1] + delta[1])
            if nbr in board:
                gph.add_edge(square, nbr)
    return gph
