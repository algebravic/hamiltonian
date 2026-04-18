#!/bin/bash
# Run diverse DP profiles for MLP training diversity
# Each job is run with the C sort-merge backend
# Edit ham_count.py path as needed

set -e
cd "$(dirname "$0")"

echo 'Running n=65 random_s83810...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 65
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_65_random_s83810.txt').read().split(',')]
import sys
print('# n = 65  ordering=random_s83810', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_65_random_s83810.txt &

echo 'Running n=65 random_s14592...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 65
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_65_random_s14592.txt').read().split(',')]
import sys
print('# n = 65  ordering=random_s14592', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_65_random_s14592.txt &

echo 'Running n=65 random_s3278...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 65
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_65_random_s3278.txt').read().split(',')]
import sys
print('# n = 65  ordering=random_s3278', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_65_random_s3278.txt &

echo 'Running n=65 bfs_v32...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 65
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_65_bfs_v32.txt').read().split(',')]
import sys
print('# n = 65  ordering=bfs_v32', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_65_bfs_v32.txt &

echo 'Running n=65 bfs_v55...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 65
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_65_bfs_v55.txt').read().split(',')]
import sys
print('# n = 65  ordering=bfs_v55', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_65_bfs_v55.txt &

echo 'Running n=68 random_s32098...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 68
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_68_random_s32098.txt').read().split(',')]
import sys
print('# n = 68  ordering=random_s32098', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_68_random_s32098.txt &

echo 'Running n=68 random_s29256...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 68
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_68_random_s29256.txt').read().split(',')]
import sys
print('# n = 68  ordering=random_s29256', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_68_random_s29256.txt &

echo 'Running n=68 random_s18289...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 68
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_68_random_s18289.txt').read().split(',')]
import sys
print('# n = 68  ordering=random_s18289', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_68_random_s18289.txt &

echo 'Running n=68 bfs_v50...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 68
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_68_bfs_v50.txt').read().split(',')]
import sys
print('# n = 68  ordering=bfs_v50', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_68_bfs_v50.txt &

echo 'Running n=68 bfs_v32...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 68
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_68_bfs_v32.txt').read().split(',')]
import sys
print('# n = 68  ordering=bfs_v32', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_68_bfs_v32.txt &

echo 'Running n=70 random_s88696...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 70
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_70_random_s88696.txt').read().split(',')]
import sys
print('# n = 70  ordering=random_s88696', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_70_random_s88696.txt &

echo 'Running n=70 random_s97080...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 70
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_70_random_s97080.txt').read().split(',')]
import sys
print('# n = 70  ordering=random_s97080', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_70_random_s97080.txt &

echo 'Running n=70 random_s71482...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 70
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_70_random_s71482.txt').read().split(',')]
import sys
print('# n = 70  ordering=random_s71482', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_70_random_s71482.txt &

echo 'Running n=70 bfs_v62...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 70
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_70_bfs_v62.txt').read().split(',')]
import sys
print('# n = 70  ordering=bfs_v62', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_70_bfs_v62.txt &

echo 'Running n=70 bfs_v39...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 70
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_70_bfs_v39.txt').read().split(',')]
import sys
print('# n = 70  ordering=bfs_v39', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_70_bfs_v39.txt &

echo 'Running n=72 random_s55302...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 72
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_72_random_s55302.txt').read().split(',')]
import sys
print('# n = 72  ordering=random_s55302', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_72_random_s55302.txt &

echo 'Running n=72 random_s4165...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 72
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_72_random_s4165.txt').read().split(',')]
import sys
print('# n = 72  ordering=random_s4165', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_72_random_s4165.txt &

echo 'Running n=72 random_s3905...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 72
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_72_random_s3905.txt').read().split(',')]
import sys
print('# n = 72  ordering=random_s3905', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_72_random_s3905.txt &

echo 'Running n=72 bfs_v31...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 72
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_72_bfs_v31.txt').read().split(',')]
import sys
print('# n = 72  ordering=bfs_v31', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_72_bfs_v31.txt &

echo 'Running n=72 bfs_v13...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 72
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_72_bfs_v13.txt').read().split(',')]
import sys
print('# n = 72  ordering=bfs_v13', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_72_bfs_v13.txt &

echo 'Running n=74 random_s30495...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 74
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_74_random_s30495.txt').read().split(',')]
import sys
print('# n = 74  ordering=random_s30495', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_74_random_s30495.txt &

echo 'Running n=74 random_s66237...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 74
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_74_random_s66237.txt').read().split(',')]
import sys
print('# n = 74  ordering=random_s66237', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_74_random_s66237.txt &

echo 'Running n=74 random_s78907...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 74
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_74_random_s78907.txt').read().split(',')]
import sys
print('# n = 74  ordering=random_s78907', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_74_random_s78907.txt &

echo 'Running n=74 bfs_v70...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 74
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_74_bfs_v70.txt').read().split(',')]
import sys
print('# n = 74  ordering=bfs_v70', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_74_bfs_v70.txt &

echo 'Running n=74 bfs_v39...'
python3 -c "
import sys; sys.path.insert(0, '.')
from ham_ordering import build_graph
from ham_dp_c import count_hamiltonian_paths_sm
n = 74
adj = build_graph(n)
order = [int(x) for x in open('profiles_diverse/order_74_bfs_v39.txt').read().split(',')]
import sys
print('# n = 74  ordering=bfs_v39', file=sys.stderr)
result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)
print(f'ham_paths={result}', file=sys.stderr)
" 2> profiles_diverse/profile_74_bfs_v39.txt &

wait
echo 'All done.'
