"""Verify the certificates behind Theorem 3.5 of doc/square_sum_bounds.tex.

For each seed in data/bounds/certificate.json (a Hamiltonian 1->3 path of
G(n0), n0 odd, plus k separated parity-preserving moves) this checks the seed,
then builds the iterated Gerbicz lifts G(n0) -> G(25 n0 + 12) -> ... and checks
at every level that the lifted path is a Hamiltonian 1->3 path and that the
25^m k lifted moves are valid and separated.

    python -m hamiltonian.bounds.certificate --levels 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lift import lift, lift_moves
from .moves import check_move, check_separated
from .squares import is_ham_path, issq

DATA = Path(__file__).resolve().parents[2] / "data" / "bounds" / "certificate.json"


def check_glue(glue, t=5):
    h = (t * t - 1) // 2
    assert glue[0] == 1 and glue[-1] == 3
    assert all(issq(glue[k] + glue[k + 1]) or (glue[k] > h and glue[k + 1] > h)
               for k in range(len(glue) - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--file", type=Path, default=DATA)
    a = ap.parse_args()
    cert = json.loads(a.file.read_text())
    glue = cert["glue"]
    check_glue(glue)
    for seed in cert["seeds"]:
        path, n = seed["path"], len(seed["path"])
        moves = [(m["i"], m["alt"]) for m in seed["moves"]]
        assert n % 2 == 1 and path[0] == 1 and path[-1] == 3 and is_ham_path(path, n)
        for i, alt in moves:
            check_move(path, i, alt)
        check_separated(moves)
        print(f"seed n0={n}: Hamiltonian 1->3 path with {len(moves)} separated moves")
        for level in range(1, a.levels + 1):
            new, offsets = lift(path, glue)
            N = 25 * n + 12
            assert path[0] == 1 and new[0] == 1 and new[-1] == 3 and is_ham_path(new, N)
            moves = lift_moves(path, moves, offsets)
            for i, alt in moves:
                check_move(new, i, alt)
            check_separated(moves)
            print(f"  level {level}: N={N}: Hamiltonian 1->3 path, "
                  f"{len(moves)} separated moves verified")
            path, n = new, N


if __name__ == "__main__":
    main()
