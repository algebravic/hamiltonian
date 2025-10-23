"""
  Utilities
"""
from typing import List, Hashable, Tuple, Iterable
from enum import Enum
import networkx as nx
from graphillion import GraphSet
from .vertex_separation import pathwidth_order

class GraphOrder(Enum):
    Default = 0
    Increasing = 1
    Decreasing = 2
    PathWidth = 3
    Dfs = 4
    Bfs = 5

ODICT = {GraphOrder.Default: "default",
         GraphOrder.Increasing: "increasing degree",
         GraphOrder.Decreasing: "decreasing degree"}

def relabeled_graph(gph: nx.Graph, order: GraphOrder, **kwds) -> nx.Graph:

    if order == GraphOrder.PathWidth:
        ngph, _, _ = pathwidth_order(gph, **kwds)
        return ngph
    else:
        return nx.convert_node_labels_to_integers(gph,
                                                  ordering = ODICT.get(order, "default"),
                                                  first_label = 1)
def use_networkx():

    GraphSet.converters['to_graph'] = nx.from_edgelist
    GraphSet.converters['to_edges'] = nx.to_edgelist

def degree_order(gph: nx.Graph, decreasing: bool = True) -> Iterable[Hashable]:

    discrim = max if decreasing else min


    xgph = gph.copy()

    nds = set(gph.nodes)

    while len(xgph.nodes) > 0:

        degs = dict(nx.degree(xgph)).items()
        nxt = discrim(degs, key=lambda _: _[1])[0]
        yield nxt
        nds.remove(nxt)
        xgph = nx.subgraph(xgph, list(nds))

def get_count(gph: nx.Graph, order: GraphOrder = GraphOrder.Increasing, **kwds):
    traversal = "as-is"
    match order:
        case GraphOrder.Increasing:
            edges = degree_order(gph, decreasing = False)
        case GraphOrder.Decreasing:
            edges = degree_order(gph, decreasing = True)
        case GraphOrder.PathWidth:
            _, edges = pathwidth_order(gph, **kwds)
        case GraphOrder.Dfs:
            edges = degree_order(gph, decreasing = True)
            traversal = "dfs"
        case GraphOrder.Bfs:
            edges = degree_order(gph, decreasing = True)
            traversal = "bfs"
        case _:
            print("Illegal order, using sorted")
            edges = sorted(gph.nodes)
    myedges = [_[:2] for _ in nx.to_edgelist(gph, nodelist = edges)]
    GraphSet.set_universe(myedges, traversal = traversal)
    return GraphSet.paths(None, None, is_hamilton=True).len()
