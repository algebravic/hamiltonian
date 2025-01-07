"""
  Command line program.
"""
import argparse
from .squares import get_square_count, GraphOrder
from .timer import Timer

def main():
    """
      The main command line.
    """
    parser = argparse.ArgumentParser(description="Hamiltonian Path Counter")
    parser.add_argument('nval', type=int, default=40,
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
    args = parser.parse_args()
    order = {
        'path_width': GraphOrder.PathWidth,
        'increasing' : GraphOrder.Increasing,
        'decreasing' : GraphOrder.Decreasing,
        'dfs' : GraphOrder.Dfs,
        'bfs' : GraphOrder.Bfs
        }
    with Timer(f'Get Square Count {args.nval} using {args.order}') as tim:
        
        res = get_square_count(args.nval,
            order = order.get(args.order, GraphOrder.PathWidth),
            verbose = args.verbose,
            minz = args.minz,
            exhaust = args.exhaust,
            adapt = args.adapt)
        print(f"Number of Hamiltonian Paths in Square Graph({args.nval}) = {res}")
    
if __name__ == '__main__':
    main()
