"""
  Each vertex has two standard fields, and 6 utility fields
  The standard fields are arcs: pointer to arcs list, and name: name of vertex
  The utility codes are S: string, I: integer, G: Graph, V: vertex, A: arc, Z: ignore

  Each arc has 3 standard fields and 2 utility fields.
  The standard fields are tip: the end of this arc, next: next arc in linked list,
  len: the length or weight of the arc.

  A graph has 7 standard fields and 6 utility fields
  standard:

  vertices: pointer to list of vertices
  n: total number of vertices
  m: total number of arcs
  id: a symbolic identification
  util_types: a representation of the utility types for graph, vertex and arc nodes
  data: an extra storage area for Arc and string storage
  aux_data: another storage area

  The first 6 characters of the util_types are for the vertex records
  The next 2 characters are for the arcs
  The last 6 are for the graph
  Default 'ZZZZZZZZZZZZZ' means no utility fields are used.
"""

from typing import Iterable, Tuple
from itertools import chain, pairwise
from pathlib import Path
import networkx as nx

def convert_to_sgb(gph: nx.Graph) -> Iterable[str]:
    """
      Convert a networkx graph to a Stanford Graph Base
      output format. Use the neighbors method, which gives the out
      neighbors if it's a digraph.
    """
    node_dict = {node: ind for ind, node in enumerate(gph.nodes)}

    n_nodes = len(gph.nodes)
    n_edges = len(gph.edges)
    t_edges = n_edges if gph.is_directed() else 2 * n_edges
    name = gph.name
    graph_util = 6 * 'Z'
    vertex_util = 6 * 'Z'
    arc_util = 2 * 'Z'
    util_types = vertex_util + arc_util + graph_util

    yield f'* GraphBase graph (util_types {util_types},{n_nodes}V,{t_edges}A)'
    yield f'"{name}",{n_nodes},{t_edges}'
    yield '* Vertices'
    arc_no = 0
    for ind, node in enumerate(gph.nodes):
        # node-id, first_arc + extra
        arcs = len(list(gph.neighbors(node)))
        nxt = '0' if arcs == 0 else f'A{arc_no}'
        yield f'"{node}",{nxt}'
        arc_no += arcs
    yield '* Arcs'
    # Each Arc
    # The arcs are a linked list
    arc_no = 0
    for node in gph.nodes:
        nbrs = list(gph.neighbors(node))
        if len(nbrs) == 0:
            continue
        weight = 1
        for tip_node in nbrs[: -1]:
            yield f'V{node_dict[tip_node]},A{arc_no+1},{weight}'
            arc_no += 1
        # last arc
        weight = 1
        yield f'V{node_dict[nbrs[-1]]},0,{weight}'
        arc_no += 1
    checksum = -1 # don't check    
    yield f'* Checksum {checksum}'
           

def write_sgb(fname: Path | str, gph: nx.Graph):

    with open(fname, 'w') as fil:
        fil.write('\n'.join(convert_to_sgb(gph)))
