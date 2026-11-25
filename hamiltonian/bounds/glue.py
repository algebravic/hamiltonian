"""Search for glue arrangements (Section 2.2 of doc/square_sum_bounds.tex).

The glue problem for parameters (t, u, w, parity of n): find a Hamiltonian
path from u to w in the graph on {1..h} U {segment endpoints}, h=(t^2-1)/2,
with edges = square sums, which uses every forced pair
(t^2 u + c, t^2 w + eps c) consecutively (eps = +1 if n odd, -1 if n even).

    python -m hamiltonian.bounds.glue --t 3 --exhaustive     # t = 3: none exist
    python -m hamiltonian.bounds.glue --t 5 --u 1 --w 3
"""
from __future__ import annotations

import argparse
import random
from typing import List, Optional

from .squares import issq


class Timeout(Exception):
    pass


def glue_search(t: int, u: int, w: int, n_odd: bool, exhaustive: bool = False,
                restarts: int = 50, budget: int = 200_000,
                seed: int = 0) -> Optional[List[int]] | bool:
    """Return a glue path, False if (exhaustively) none exists, None if unknown."""
    T, h = t * t, (t * t - 1) // 2
    partner = {}
    for c in range(-h, h + 1):
        a, b = T * u + c, T * w + (c if n_odd else -c)
        partner[a], partner[b] = b, a
    verts = list(range(1, h + 1)) + list(partner)
    if len(set(verts)) != len(verts):
        return False
    adj = {v: [x for x in verts if x != v and issq(x + v) and partner.get(v) != x]
           for v in verts}
    rng = random.Random(seed)
    for _ in range(1 if exhaustive else restarts):
        path, used, steps = [u], {u}, [0]

        def rec(v: int) -> bool:
            steps[0] += 1
            if not exhaustive and steps[0] > budget:
                raise Timeout
            if len(path) == len(verts):
                return v == w
            if v in partner and partner[v] not in used:      # traverse segment
                x = partner[v]
                used.add(x); path.append(x)
                if rec(x):
                    return True
                path.pop(); used.discard(x)
                return False
            cand = [x for x in adj[v]
                    if x not in used and (x != w or len(path) == len(verts) - 1)]
            if not exhaustive:
                rng.shuffle(cand)
            for x in cand:
                used.add(x); path.append(x)
                if rec(x):
                    return True
                path.pop(); used.discard(x)
            return False

        try:
            if rec(u):
                return path
            if exhaustive:
                return False
        except Timeout:
            pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", type=int, default=5)
    ap.add_argument("--u", type=int)
    ap.add_argument("--w", type=int)
    ap.add_argument("--exhaustive", action="store_true")
    a = ap.parse_args()
    h = (a.t * a.t - 1) // 2
    pairs = ([(a.u, a.w)] if a.u else
             [(u, w) for u in range(1, h + 1) for w in range(1, h + 1) if u != w])
    for u, w in pairs:
        for n_odd in (True, False):
            r = glue_search(a.t, u, w, n_odd, exhaustive=a.exhaustive)
            tag = "found" if r else ("none" if r is False else "unknown")
            print(f"t={a.t} u={u} w={w} n {'odd' if n_odd else 'even'}: {tag}"
                  + (f"  {r}" if r else ""))


if __name__ == "__main__":
    main()
