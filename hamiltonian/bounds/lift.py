"""Gerbicz's t^2-lift (Section 2 of doc/square_sum_bounds.tex).

Given a Hamiltonian path b of G(n) from u to w and a glue arrangement, build a
Hamiltonian path of G(t^2 n + h), h = (t^2-1)/2.  For c in [-h, h] the lifted
segment is T(c) = (t^2 b_p + (-1)^p c)_p  (0-based p), running from t^2 u + c
to t^2 w + (-1)^(n-1) c.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from .moves import Move


def segments(b: Sequence[int], t: int) -> Dict[int, List[int]]:
    T, h = t * t, (t * t - 1) // 2
    return {c: [T * b[p] + (c if p % 2 == 0 else -c) for p in range(len(b))]
            for c in range(-h, h + 1)}


def lift(b: Sequence[int], glue: Sequence[int], t: int = 5):
    """Return (path, offsets) where offsets[c] = (start position, reversed?)."""
    h = (t * t - 1) // 2
    segs = segments(b, t)
    by_start = {s[0]: c for c, s in segs.items()}
    by_end = {s[-1]: c for c, s in segs.items()}
    out: List[int] = []
    offsets = {}
    k = 0
    while k < len(glue):
        v = glue[k]
        if v <= h:
            out.append(v)
            k += 1
            continue
        w = glue[k + 1]
        if v in by_start and segs[by_start[v]][-1] == w:
            c = by_start[v]
            offsets[c] = (len(out), False)
            out += segs[c]
        else:
            c = by_end[v]
            assert segs[c][0] == w
            offsets[c] = (len(out), True)
            out += segs[c][::-1]
        k += 2
    return out, offsets


def lift_moves(b: Sequence[int], moves: List[Move], offsets, t: int = 5) -> List[Move]:
    """Lift each base move into every segment T(c) (Lemma 3.3)."""
    n = len(b)
    out: List[Move] = []
    for i, alt in moves:
        for c, (o, rev) in offsets.items():
            new = [t * t * alt[q] + (c if (i + q) % 2 == 0 else -c)
                   for q in range(len(alt))]
            if not rev:
                out.append((o + i, new))
            else:
                j = i + len(alt) - 1
                out.append((o + (n - 1 - j), new[::-1]))
    return out
