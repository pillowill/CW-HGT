"""Seeding and the causal-walk conditioning helpers.

``cal_conUs`` and ``get_perturbation`` are reproduced verbatim from the
reference implementation: they draw from the *global* NumPy RNG, so the order in
which they are called (and any prior seeding) determines the sampled
conditioning vector. Do not make them use an ``np.random.Generator`` without
re-checking reproducibility against the reported numbers.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = [
    "set_random_seed",
    "seed_everything",
    "cal_conUs",
    "get_perturbation",
]


def set_random_seed(seed: int) -> None:
    """Seed PyTorch (CPU + all CUDA devices), matching the original scripts."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch.

    The entry scripts only seeded PyTorch; NumPy was seeded implicitly inside
    :func:`cal_conUs`. Seeding NumPy here makes a run reproducible from the very
    first call instead of from the first fold, which is what the fold loop
    already assumed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_random_seed(seed)


def cal_conUs(num_nodes: int, seed: int) -> np.ndarray:
    """Draw the conditioning variable ``U`` in ``{1, 2}`` for every node.

    ``1`` means "feature perturbed", ``2`` means "feature kept"; see
    :func:`get_perturbation`.
    """
    np.random.seed(seed)
    return np.random.randint(1, 3, num_nodes)


def get_perturbation(con_Us, X):
    """Apply the observation mask of the backdoor adjustment to ``X``.

    Rows with ``con_Us == 1`` are the intervened ones: each dimension is kept
    with probability 0.7 and otherwise flipped (0 <-> 1). Rows with
    ``con_Us == 2`` are passed through unchanged.
    """
    x_gauss = np.zeros_like(X)
    obs = np.random.binomial(1, 0.7, X.shape[1])
    obs_x = np.zeros((1, X.shape[1]))
    for i, val in enumerate(con_Us):
        if con_Us[i] == 1:
            for j, val1 in enumerate(X[i, :]):
                if obs[j] == 1:
                    obs_x[0][j] = val1
                elif obs[j] == 0:
                    if val1 == 1:
                        obs_x[0][j] = 0
                    elif val1 == 0:
                        obs_x[0][j] = 1
            x_gauss[i] = obs_x
        elif con_Us[i] == 2:
            x_gauss[i] = X[i, :]
    return x_gauss
