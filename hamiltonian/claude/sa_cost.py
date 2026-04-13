"""
sa_cost.py
----------
Secondary cost function for the Simulated Annealing Hamiltonian-path search.

Predicts log(so/si) — the log ratio of output to input states after a DP step.
Lower values are better (fewer states = cheaper DP).

Parameters are loaded from sa_cost_params.json, which is written by
profile_analysis.py.  The JSON format is a feature-vector model:

    {
      "name":     "<model name>",
      "features": ["const", "fs", "delta_fs", "n_back", "e_bag",
                   "fs_nb", "fs_eb", "nb_eb", "nb_sq", "eb_sq"],
      "coef":     [ ... ],
      "r2":       ...,
      "rmse":     ...,
      "n_obs":    ...
    }

Supported feature names
-----------------------
  const    — intercept (always 1)
  fs       — frontier size after this step
  delta_fs — change in frontier size (fs_after - fs_before)
  n_back   — back-edges from the newly placed vertex into the frontier
  e_bag    — edges inside the frontier (bag edges)
  fs_nb    — fs * n_back
  fs_eb    — fs * e_bag
  nb_eb    — n_back * e_bag
  nb_sq    — n_back²
  eb_sq    — e_bag²
  fs_sq    — fs²

To update the model: re-run profile_analysis.py (it overwrites sa_cost_params.json)
then call sa_cost.reload() or restart the process.
"""

import json
import pathlib

_PARAMS_FILE = pathlib.Path(__file__).parent / "sa_cost_params.json"


class _Model:
    __slots__ = ("features", "coef", "name", "r2", "rmse", "n_obs")
    def __init__(self, d):
        self.features = d["features"]
        self.coef     = d["coef"]
        self.name     = d.get("name", "")
        self.r2       = d.get("r2", float("nan"))
        self.rmse     = d.get("rmse", float("nan"))
        self.n_obs    = d.get("n_obs", 0)


def _load() -> _Model:
    with open(_PARAMS_FILE) as f:
        return _Model(json.load(f))


_M: _Model = _load()


def _eval(m: _Model, fs: float, delta_fs: float, n_back: int, e_bag: int) -> float:
    """Evaluate the linear model for the given raw features."""
    feat_map = {
        "const":    1.0,
        "fs":       fs,
        "delta_fs": delta_fs,
        "n_back":   n_back,
        "e_bag":    e_bag,
        "fs_nb":    fs * n_back,
        "fs_eb":    fs * e_bag,
        "nb_eb":    n_back * e_bag,
        "nb_sq":    n_back ** 2,
        "eb_sq":    e_bag ** 2,
        "fs_sq":    fs ** 2,
    }
    return sum(c * feat_map[f] for c, f in zip(m.coef, m.features))


def sa_cost(fs: float, delta_fs: float, n_back: int, e_bag: int) -> float:
    """
    Predict log(so/si) for a DP step characterised by:

        fs       -- frontier size after placing this vertex
        delta_fs -- change in frontier size (fs_after - fs_before)
        n_back   -- back-edges from the newly placed vertex into the frontier
        e_bag    -- edges entirely within the frontier (bag edges)

    Returns a float.  Use math.exp(sa_cost(...)) to get the multiplicative
    expansion factor for the SA proxy.

    Parameters are read from sa_cost_params.json; call reload() after
    updating that file without restarting the process.
    """
    return _eval(_M, fs, delta_fs, n_back, e_bag)


def reload() -> None:
    """Hot-reload parameters from sa_cost_params.json."""
    global _M
    _M = _load()


def model_info() -> dict:
    """Return a summary of the currently loaded model."""
    return {
        "name":     _M.name,
        "features": _M.features,
        "coef":     _M.coef,
        "r2":       _M.r2,
        "rmse":     _M.rmse,
        "n_obs":    _M.n_obs,
    }
