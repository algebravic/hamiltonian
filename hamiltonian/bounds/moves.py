"""Parity-preserving moves (Definition 3.1 of doc/square_sum_bounds.tex).

A move on a Hamiltonian path P = v_0 ... v_{N-1} is a window [i, j] with
1 <= i <= j <= N-2 and a different ordering w_i ... w_j of {v_i,...,v_j} such
that v_{i-1} w_i ... w_j v_{j+1} is a path and every vertex keeps the parity
of its position.  Positions are 0-based here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .squares import issq

Move = Tuple[int, List[int]]          # (start position i, new window content)


def alternative(path: Sequence[int], i: int, j: int,
                adj: Dict[int, List[int]]) -> Optional[List[int]]:
    """A parity-preserving rerouting of window [i, j], or None."""
    window = set(path[i:j + 1])
    par = {path[p]: p % 2 for p in range(i, j + 1)}
    target, length, orig = path[j + 1], j - i + 1, list(path[i:j + 1])
    seq: List[int] = []
    used: set = set()

    def rec(v: int) -> Optional[List[int]]:
        if len(seq) == length:
            return list(seq) if issq(v + target) and seq != orig else None
        pos = i + len(seq)
        for u in adj[v]:
            if u in window and u not in used and par[u] == pos % 2:
                used.add(u)
                seq.append(u)
                r = rec(u)
                if r:
                    return r
                seq.pop()
                used.discard(u)
        return None

    return rec(path[i - 1])


def shortest_windows(path: Sequence[int], adj, max_len: int) -> List[Tuple[int, int]]:
    """For each start i, the shortest window [i, j] (length <= max_len) admitting a move."""
    n, res = len(path), []
    for i in range(1, n - 1):
        for j in range(i, min(n - 1, i + max_len)):
            if alternative(path, i, j, adj) is not None:
                res.append((i, j))
                break
    return res


def max_separated(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Maximum family of windows that are disjoint and separated by >= 1
    position (greedy by right endpoint, which is optimal for intervals)."""
    chosen, last = [], -10
    for i, j in sorted(windows, key=lambda w: w[1]):
        if i >= last + 2:
            chosen.append((i, j))
            last = j
    return chosen


def check_move(path: Sequence[int], i: int, alt: Sequence[int]) -> None:
    """Assert that (i, alt) is a parity-preserving move of `path`."""
    j = i + len(alt) - 1
    assert 1 <= i and j <= len(path) - 2, "window not interior"
    win = list(path[i:j + 1])
    assert sorted(win) == sorted(alt) and win != list(alt)
    par = {path[p]: p % 2 for p in range(i, j + 1)}
    assert all(par[alt[q]] == (i + q) % 2 for q in range(len(alt))), "parity"
    q = [path[i - 1], *alt, path[j + 1]]
    assert all(issq(q[k] + q[k + 1]) for k in range(len(q) - 1)), "edge"


def check_separated(moves: List[Move]) -> None:
    ms = sorted(moves)
    for (i1, a1), (i2, _) in zip(ms, ms[1:]):
        assert i2 >= i1 + len(a1) + 1, "moves not separated"
