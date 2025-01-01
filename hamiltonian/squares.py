"""A090460:
  
  1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 10, 12, 35, 52, 19, 20, 349, 361,
  637, 3678, 15237, 11875, 13306, 10964, 27223, 37054, 201408, 510152,
  1995949, 4867214, 11255174, 35705858, 63029611, 129860749,
  258247089, 190294696, 686125836, 2195910738, 5114909395, 9141343219,
  19769529758, 44678128099, 63885400119


A071983

  Square chains: the number of permutations (reversals not counted as
  different) of the numbers 1 to n such that the sum of any two
  consecutive numbers is a square.


  1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 10, 12, 35, 52, 19, 20, 349, 392, 669,
  4041, 17175, 12960, 14026, 11889, 29123, 39550, 219968, 553694,
  2178103, 5301127, 12220138, 38838893, 68361609, 140571720,
  280217025, 204853870, 738704986, 2368147377, 5511090791, 9802605881,
  21164463050, 47746712739, 68092497615, 123092214818

"""
from typing import List
from math import floor, sqrt, ceil
import networkx as nx
from graphillion import GraphSet
from .util import get_count, GraphOrder

def square_graph(num: int) -> nx.Graph:
    """
      Input: n a positive integer
      Ouput: A labeled undirected graph with labels from 1 to n
      (i,j) is an edge if i+j is a non-zero square

      Method:
      For each i in [1,...,n] consider the possible n >= j > i.

      If n >= j >= i+1, 2i+1 <= r^2 = i+j <= i + n
      Note: i + j = r^2  <==> 0 < r < floor(sqrt(2 * n - 1))
      
    """
    gph = nx.Graph()
    squares = {_ ** 2 for _ in range(2, floor(2 * num - 1))}
    gph.add_edges_from(((ind, jind)
                        for ind in range(1, num)
                        for jind in range(ind + 1, num + 1)
                        if ind + jind in squares))
    return gph

def get_square_count(num: int, order: GraphOrder = GraphOrder.Decreasing, **kwds):

    return get_count(square_graph(num), order = order, **kwds)

def square_sequence(gph: nx.Graph) -> List[int]:

    path = []
    last = list(nx.neighbors(gph, 0))[0] # Only 1
    prev = 0
    # Nodes are 0, 1, ..., n+1, so len(gph) = n+2
    while True:
        path.append(last)
        nxt = [_ for _ in nx.neighbors(gph, last) if _ != prev]
        if nxt:
            prev, last = last, nxt[0]
        else:
            break
    return path[:-1]
