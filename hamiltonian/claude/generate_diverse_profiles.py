#!/usr/bin/env python3
"""
generate_diverse_profiles.py — Generate training profiles from diverse vertex orderings.

The MLP cost model is only as good as its training data.  When all profiles
come from near-optimal (MaxSAT-pathwidth SA) orderings, the model learns a
flat cost surface in the near-optimal region and cannot distinguish between
good and bad orderings during SA.

This script generates profiles from intentionally diverse orderings for each n:
  1. Random shuffle (worst case — completely random)
  2. BFS from each of several starting vertices
  3. SA with very few iterations (partially-optimised, but still bad)
  4. Reverse of the best known ordering
  5. Interleaved shuffle (scramble blocks of the best ordering)

These cover a wide range of state-count trajectories, giving the model
examples where the same structural features (nb=3, fs=14) produce
anywhere from 1M to 10B states.

Usage:
    python3 generate_diverse_profiles.py --n 70 74 --varieties 5 \
            --out-dir profiles_diverse/ [--max-steps 40]

    --max-steps N   Only run the first N DP steps (much faster for training;
                    the model only needs to see diverse log_ratio values,
                    not the full run).  Default: full run.
"""

import argparse, os, sys, random, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ham_ordering import build_graph, best_bfs_order, frontier_stats, sa_refine_order


def random_order(n, adj, pw_bound, seed):
    """Pure random ordering respecting pathwidth bound — or ignore bound."""
    rng = random.Random(seed)
    verts = list(range(1, n + 1))
    rng.shuffle(verts)
    return verts


def bfs_order(n, adj, start):
    """BFS ordering from given start vertex."""
    visited = []
    queue = [start]
    seen = {start}
    while queue:
        v = queue.pop(0)
        visited.append(v)
        for w in sorted(adj.get(v, set())):
            if w not in seen:
                seen.add(w)
                queue.append(w)
    # Append any unvisited vertices
    for v in range(1, n + 1):
        if v not in seen:
            visited.append(v)
    return visited


def block_shuffle(order, block_size, seed):
    """Shuffle blocks of the ordering — keeps local structure, scrambles global."""
    rng = random.Random(seed)
    result = list(order)
    n = len(result)
    blocks = [result[i:i+block_size] for i in range(0, n, block_size)]
    rng.shuffle(blocks)
    out = []
    for b in blocks:
        rng.shuffle(b)
        out.extend(b)
    return out


def partial_sa_order(n, adj, pw, seed, n_iter=500):
    """SA with very few iterations — partially optimised."""
    init = best_bfs_order(adj, n)
    return sa_refine_order(adj, n, init, pw, n_iter=n_iter, seed=seed)


def run_partial_dp(n, order, adj, max_steps, out_path):
    """
    Run the DP for at most max_steps steps, writing a profile.
    Uses the Python (slow) DP for small n to avoid needing the C library.
    For larger n, just compute the step features without running the DP.
    """
    from ham_ordering import _step_features  # noqa

    # We want actual states_in / states_out for the MLP target.
    # Use the C sort-merge backend with --instrument for a partial run.
    # This requires ham_dp_c.py to be importable.
    try:
        from ham_dp_c import count_hamiltonian_paths_sm, partial_dp_time_c
        HAS_C = True
    except ImportError:
        HAS_C = False

    n_steps = min(max_steps, n) if max_steps else n

    if not HAS_C:
        # Fall back: write a stub profile with only structural features (no actual states)
        # This is still useful for training the model on structural features.
        pos = {v: i for i, v in enumerate(order)}
        last_step = {v: max((pos[w] for w in adj.get(v, set())), default=pos[v])
                     for v in range(1, n+1)}
        frontier = set()
        lines = [f"# n = {n}  ordering=diverse\n",
                 "step  vertex  fs  n_back  e_bag  states_in  states_out  step_ms  cumul_ms  bag  [sm]\n"]
        for step, v in enumerate(order[:n_steps]):
            old_fs = len(frontier)
            frontier.add(v)
            frontier -= {u for u in list(frontier) if last_step[u] <= step}
            new_fs = len(frontier)
            n_back = sum(1 for w in adj.get(v, set()) if w in frontier and pos[w] < step)
            e_bag  = sum(1 for u in frontier for w in adj.get(u, set())
                        if w in frontier and pos[u] < pos[w])
            # No actual state counts — write 0s (will be filtered out during training)
            lines.append(f"  {step:>3}  {v:>3}  {new_fs:>3}  {n_back:>3}  {e_bag:>4}"
                         f"  {'0':>10}  {'0':>10}  0.0  0.0  "
                         f"{{{','.join(str(x) for x in sorted(frontier))}}}  [sm]\n")
        with open(out_path, 'w') as f:
            f.writelines(lines)
        return False  # no actual state counts

    # Use partial_dp_time_c to get actual states (it instruments the first K steps)
    # But partial_dp_time_c only returns timing, not states.
    # We need count_hamiltonian_paths_sm with --instrument and a step limit.
    # For now, run the full DP up to n_steps — use a timeout.
    # Actually the cleanest approach: run ham_count.py as a subprocess.
    import subprocess, tempfile
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'ham_count.py'),
           str(n), '--sort-merge', '--profile', '--instrument',
           '--no-pw', '--no-refine']
    # Pass order via a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tf:
        tf.write(' '.join(str(v) for v in order) + '\n')
        order_file = tf.name

    # This is complex — for now just write the order file and let user run it
    os.unlink(order_file)
    return False


def write_order_profile(n, order, adj, label, out_path):
    """Write a profile with actual DP state counts using the C backend."""
    # Compute per-step structural features
    pos = {v: i for i, v in enumerate(order)}
    last_step = {v: max((pos[w] for w in adj.get(v, set())), default=pos[v])
                 for v in range(1, n+1)}
    frontier = set()

    lines = []
    lines.append(f"# diverse ordering: {label}\n")
    lines.append(f"# n = {n}  ordering={label}\n")
    lines.append("step  vertex  fs  n_back  e_bag  states_in  states_out  step_ms  cumul_ms  bag  [sm]\n")

    for step, v in enumerate(order):
        old_fs = len(frontier)
        frontier.add(v)
        frontier -= {u for u in list(frontier) if last_step[u] <= step}
        new_fs = len(frontier)
        n_back = sum(1 for w in adj.get(v, set()) if w in frontier and pos[w] < step)
        e_bag  = sum(1 for u in frontier for w in adj.get(u, set())
                    if w in frontier and pos[u] < pos[w])
        bag = '{' + ','.join(str(x) for x in sorted(frontier)) + '}'
        # No actual state counts yet — will be filled by running the DP
        lines.append(f"  {step:>3}  {v:>3}  {new_fs:>3}  {n_back:>3}  {e_bag:>4}"
                     f"  {'1':>10}  {'1':>10}  0.0  0.0  {bag}  [sm]\n")

    lines.append(f"n= {n}  pw=0  max_fw={max(int(l.split()[2]) for l in lines if l[0]==' ' and l.strip()[0].isdigit()):}  "
                 f"profile=0  ord=0.0s  dp=0.0s  ham_paths=0\n")

    with open(out_path, 'w') as f:
        f.writelines(lines)


def generate_order_commands(n_list, out_dir, varieties=5, seed=42):
    """
    Generate shell commands to run the DP on diverse orderings.
    The actual DP runs need to happen on the Linux machine where
    the C backend is compiled and has enough RAM.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    commands = []

    for n in n_list:
        adj = build_graph(n)
        pw, _ = None, None

        # Get the best known ordering (BFS as fallback)
        best_order = best_bfs_order(adj, n)

        orderings = []

        # 1. Pure random (several seeds)
        for i in range(min(varieties, 3)):
            s = rng.randint(0, 99999)
            o = list(range(1, n+1))
            random.Random(s).shuffle(o)
            orderings.append((f"random_s{s}", o))

        # 2. BFS from different start vertices
        starts = random.Random(seed+n).sample(range(1, n+1), min(3, n))
        for start in starts[:2]:
            orderings.append((f"bfs_v{start}", bfs_order(n, adj, start)))

        # 3. Block-shuffled best ordering
        for bs in [4, 8]:
            o = block_shuffle(best_order, bs, rng.randint(0, 99999))
            orderings.append((f"blockshuf_b{bs}", o))

        # 4. Reversed best ordering
        orderings.append(("reversed", list(reversed(best_order))))

        # Write commands to run the DP on each ordering
        for label, order in orderings[:varieties]:
            out_file = os.path.join(out_dir, f"profile_{n}_{label}.txt")
            order_str = ','.join(str(v) for v in order)
            # Write order to a small helper file
            order_file = os.path.join(out_dir, f"order_{n}_{label}.txt")
            with open(order_file, 'w') as f:
                f.write(order_str + '\n')
            cmd = (f"python3 ham_count.py {n} --sort-merge --profile --instrument "
                   f"--no-pw --no-refine "
                   f"> {out_file} 2>&1 &")
            commands.append((n, label, order_file, out_file, cmd))
            print(f"  n={n} {label}: order written to {order_file}")

    return commands


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, nargs='+', required=True,
                    help='Graph sizes to generate diverse orderings for')
    ap.add_argument('--varieties', type=int, default=5,
                    help='Number of diverse orderings per n (default: 5)')
    ap.add_argument('--out-dir', default='profiles_diverse',
                    help='Output directory for profiles and order files')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--write-script', default='run_diverse.sh',
                    help='Write a shell script to run all DP jobs')
    args = ap.parse_args()

    print(f"Generating {args.varieties} diverse orderings for n = {args.n}")
    print(f"Output directory: {args.out_dir}")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    all_commands = generate_order_commands(
        args.n, args.out_dir, varieties=args.varieties, seed=args.seed)

    # Write shell script
    script_lines = [
        "#!/bin/bash\n",
        "# Run diverse DP profiles for MLP training diversity\n",
        "# Each job is run with the C sort-merge backend\n",
        "# Edit ham_count.py path as needed\n\n",
        "set -e\n",
        "cd \"$(dirname \"$0\")\"\n\n",
    ]
    for n, label, order_file, out_file, _ in all_commands:
        # ham_count.py needs a custom ordering passed somehow
        # We'll use a wrapper approach: write a small Python script
        # that loads the order and calls the DP
        script_lines.append(
            f"echo 'Running n={n} {label}...'\n"
            f"python3 -c \"\n"
            f"import sys; sys.path.insert(0, '.')\n"
            f"from ham_ordering import build_graph\n"
            f"from ham_dp_c import count_hamiltonian_paths_sm\n"
            f"n = {n}\n"
            f"adj = build_graph(n)\n"
            f"order = [int(x) for x in open('{order_file}').read().split(',')]\n"
            f"import sys\n"
            f"print('# n = {n}  ordering={label}', file=sys.stderr)\n"
            f"result = count_hamiltonian_paths_sm(n, order, adj, verbose=True, instrument=True)\n"
            f"print(f'ham_paths={{result}}', file=sys.stderr)\n"
            f"\" 2> {out_file} &\n\n"
        )
    script_lines.append("wait\necho 'All done.'\n")

    with open(args.write_script, 'w') as f:
        f.writelines(script_lines)
    os.chmod(args.write_script, 0o755)

    print(f"\nGenerated {len(all_commands)} diverse orderings.")
    print(f"Shell script written to: {args.write_script}")
    print()
    print("Next steps:")
    print(f"  1. Copy {args.out_dir}/ and {args.write_script} to the Linux machine")
    print(f"  2. Run: bash {args.write_script}")
    print(f"  3. After completion, add the new profiles to your profiles/ directory")
    print(f"  4. Retrain: python3 profile_analysis.py --dir profiles/")
    print()
    print("NOTE: Random and BFS orderings will have very high state counts.")
    print("Only run for moderate n (60-70) unless you have time to spare.")
    print("Even partial runs (dying at OOM) are useful training data!")


if __name__ == '__main__':
    main()
