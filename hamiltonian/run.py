"""
  Command line program.
"""
import argparse
import networkx as nx
from .util import GraphOrder, get_count, use_networkx, relabeled_graph
from .squares import square_graph
from .knights import knight_graph
from .hamming import hamming_graph
from .grid import grid_graph
from .timer import Timer
from .to_sgb import write_sgb

def write_graph():
    """
      Write out the selected graph in GB format.
    """
    parser = argparse.ArgumentParser(description="Graph generator")
    parser.add_argument('graph', type=str, default='square',
                        help='The graph family')
    parser.add_argument('args', type=int, nargs='*',
                        help='The parameters for the family')
    parser.add_argument('--dir', type=str, default='',
                        help='The directory to write the file')
    parser.add_argument('--path', type=bool, default=False,
                        help='If True create the path graph')

    families = {'knight': (knight_graph, 2),
                'square': (square_graph, 1),
                'hamming': (hamming_graph, 1),
                'grid'   : (grid_graph, 2)
                }
    args = parser.parse_args()
    if args.graph not in families:
        print(f'Unknown graph family: {args.graph}')
        return
    fcn, nargs = families[args.graph]
    if nargs > len(args.args):
        print(f'Extra arguments ignored: {args.args[nargs: ]}')
    gph = fcn(*args.args[: nargs])
    base = args.graph
    if args.path:
        base += '_path'
    gph.add_edges_from([('root', _) for _ in gph.nodes])
    name = base + '_' + '_'.join(map(str, args.args)) + '.gb'
    write_sgb(args.dir + name, gph)

def run_tours():
    """
      The main command line.
    """
    parser = argparse.ArgumentParser(description="Hamiltonian Path Counter")
    parser.add_argument('graph', type=str, default='square',
                        help='The graph family')
    parser.add_argument('nval', type=int, default=[40], nargs = '*',
                        help='The parameter for the graph family')
    parser.add_argument('--order', type=str, default='path_width',
                        help='The edge ordering algorithm')
    parser.add_argument('--verbose', type=int, default=0,
                        help='The verbosity level for RC2')
    parser.add_argument('--adapt', type=bool, default=False,
                        help='Whether to use the adapt option for RC2')
    parser.add_argument('--minz', type=bool, default=False,
                        help='Whether to use the minz option for RC2')
    parser.add_argument('--exhaust', type=bool, default=False,
                        help='Whether to use the exhaust option for RC2')
    parser.add_argument('--stratified', type=bool, default=False,
                        help='Whether to use the stratified solver')
    parser.add_argument('--type', type=str, default = 'paths',
                        help='Hamiltonian cycles or paths')
    info = {'square' : ('Square', square_graph, 1),
            'knight' : ('Knight', knight_graph, 2),
            'hamming' : ('Hamming', hamming_graph, 1),
            'grid'    : ('Grid', grid_graph, 2)
            }
    args = parser.parse_args()
    name, fcn, nargs = info.get(args.graph, info['square'])
    cycles = args.type == 'cycles'
    kind = 'Cycles' if cycles else 'Paths'
    fargs = args.nval[: nargs]
    order = {
        'path_width': GraphOrder.PathWidth,
        'increasing' : GraphOrder.Increasing,
        'decreasing' : GraphOrder.Decreasing,
        'dfs' : GraphOrder.Dfs,
        'bfs' : GraphOrder.Bfs
        }
    # use_networkx()
    with Timer(f'Get {name} Count {args.nval} using {args.order}') as tim:
        gph = relabeled_graph(fcn(*fargs), GraphOrder.Default)
        res = get_count(gph,
                        cycles = cycles,
                        order = order.get(args.order, GraphOrder.PathWidth),
                        verbose = args.verbose,
                        minz = args.minz,
                        exhaust = args.exhaust,
                        adapt = args.adapt,
                        stratified = args.stratified)
        print(f"Number of Hamiltonian {kind} in {name} Graph{tuple(fargs)} = {res}")
    
if __name__ == '__main__':
    run_tours()
