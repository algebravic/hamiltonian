"""Basic utilities for the square-sum graph G(n)."""
from __future__ import annotations

import math
import random
import sys
from typing import Dict, List, Optional

sys.setrecursionlimit(1_000_000)


def issq(x: int) -> bool:
    """True iff x is a perfect square (x >= 0)."""
    if x < 0:
        return False
    r = math.isqrt(x)
    return r * r == x


def adjacency(n: int) -> Dict[int, List[int]]:
    """Adjacency lists of G(n)."""
    return {v: [u for u in range(1, n + 1) if u != v and issq(u + v)]
            for v in range(1, n + 1)}


def is_ham_path(path: List[int], n: int) -> bool:
    """Check that `path` is a Hamiltonian path of G(n)."""
    return (sorted(path) == list(range(1, n + 1))
            and all(issq(path[k] + path[k + 1]) for k in range(n - 1)))


def all_paths(n: int, s: int, t: int) -> List[List[int]]:
    """All Hamiltonian paths of G(n) from s to t (exhaustive; small n only)."""
    adj = adjacency(n)
    out: List[List[int]] = []
    path, used = [s], {s}

    def rec(v: int) -> None:
        if len(path) == n:
            if v == t:
                out.append(path.copy())
            return
        for u in adj[v]:
            if u not in used and (u != t or len(path) == n - 1):
                used.add(u)
                path.append(u)
                rec(u)
                path.pop()
                used.discard(u)

    rec(s)
    return out


def random_path(n: int, s: int, t: int, adj=None, tries: int = 200,
                budget: int = 200_000) -> Optional[List[int]]:
    """Randomized Warnsdorff-style DFS for a Hamiltonian s-t path of G(n)."""
    adj = adj or adjacency(n)
    for _ in range(tries):
        path, used, steps = [s], {s}, [0]

        def rec(v: int) -> bool:
            steps[0] += 1
            if steps[0] > budget:
                return False
            if len(path) == n:
                return v == t
            cand = [u for u in adj[v]
                    if u not in used and (u != t or len(path) == n - 1)]
            random.shuffle(cand)
            cand.sort(key=lambda u: sum(1 for w in adj[u] if w not in used))
            for u in cand:
                used.add(u)
                path.append(u)
                if rec(u):
                    return True
                path.pop()
                used.discard(u)
            return False

        if rec(s):
            return path
    return None
