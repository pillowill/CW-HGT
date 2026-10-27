"""Walk-baseline comparison (the retired ``CW_ays.py``).

Compares the causal walk against the random-walk baselines (``RW``, ``N2VRW``) by
turning each sampled walk set into an adjacency matrix and scoring it against the
ground-truth association matrix restricted to lncRNA-stress links.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from cw_hgt.utils import cal_conUs, save_causal_walks
from cw_hgt.walk import causal as CW

__all__ = [
    "WALK_METRIC_NAMES",
    "read_json2arr",
    "walk_metrics",
    "sample_walks",
    "evaluate_walk_baselines",
]

WALK_METRIC_NAMES: Sequence[str] = (
    "precision", "recall", "accuracy", "f1", "ap",
    "precision_at_k", "recall_at_k", "hits_at_k", "specificity",
)


def read_json2arr(json_file: str) -> np.ndarray:
    """Read a sampled-walk JSON dump and return its adjacency matrix.

    The dump maps ``str(node) -> list of neighbours``; for multi-walk dumps the
    first walk of each node is used.
    """
    with open(json_file, encoding="utf-8") as fh:
        adjacency_list = json.load(fh)

    nodes = list(adjacency_list.keys())
    node_index = {int(node): index for index, node in enumerate(nodes)}
    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for node, neighbours in adjacency_list.items():
        if isinstance(neighbours[0], list):
            neighbours = neighbours[0]
        for neighbour in neighbours:
            if neighbour != int(node):
                adjacency_matrix[node_index[int(node)]][node_index[neighbour]] = 1
    return adjacency_matrix


def walk_metrics(true_matrix: np.ndarray, pred_matrix: np.ndarray, k: int = 10) -> np.ndarray:
    """Binary metrics over a balanced subsample of the two matrices.

    The subsample is drawn with the global NumPy RNG (as in the original), so
    seed before calling if you need a stable value.
    """
    pred_flat = pred_matrix.flatten()
    true_flat = true_matrix.flatten()

    positive = np.where(true_flat == 1)[0]
    negative = np.where(true_flat == 0)[0]
    n = min(len(positive), len(negative))
    chosen = np.concatenate((
        np.random.choice(positive, n, replace=False),
        np.random.choice(negative, n, replace=False),
    ))
    np.random.shuffle(chosen)

    pred = pred_flat[chosen]
    true = true_flat[chosen]

    order = np.argsort(-pred)
    top_k = order[:k]
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    tn = np.sum((pred == 0) & (true == 0))
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return np.array([
        precision_score(true, pred, zero_division=0),
        recall_score(true, pred, zero_division=0),
        accuracy_score(true, pred),
        f1_score(true, pred, zero_division=0),
        average_precision_score(true, pred),
        np.sum(true[top_k]) / k,
        np.sum(true[top_k]) / np.sum(true) if np.sum(true) else 0.0,
        np.sum(true[top_k]),
        specificity,
    ])


def sample_walks(adj: np.ndarray, con_us: np.ndarray, walk_type: str, p: float, q: int,
                 walk_length: int, window_size: int, multi_n: int = 1):
    """Sample walks of the requested kind and return ``(list, dict)``."""
    if walk_type == "CW":
        sampler = CW.MultiCausalWalk(p, q, walk_length, window_size, multi_n)
        return sampler(adj, con_us)
    n_nodes = adj.shape[0]
    if walk_type == "RW":
        return CW.RandomWalk(adj, n_nodes, walk_length).random_walks()
    return CW.N2V_RandomWalk(adj, n_nodes, p, q, walk_length).node2vec_random_walks()


def evaluate_walk_baselines(dataset, hg, walk_type: str, *, walk_length: int = 50,
                            p: float = 4, q: int = 3, folds: int = 10, seed: int = 3047,
                            out_dir: str | None = None, k: int = 10) -> Dict[str, float]:
    """Run the baseline comparison for one ``walk_type``.

    Mirrors ``CW_ays.py``: the ground truth is the lncRNA-stress block of the
    graph adjacency, and the prediction is the same block of the walk-derived
    adjacency. Sampled walks are cached in ``out_dir`` when given.

    Returns ``{"mean": {metric: value}, "std": {...}}``.
    """
    from cw_hgt.constants import get_task

    task = get_task("lnc_mi_stress_graphs")  # the baseline script was 3-type only
    adj = dataset.get_adj(hg)
    n_lnc = hg.num_nodes("lnc")
    n_mi = hg.num_nodes("mi")
    true_block = adj[:n_lnc, n_lnc + n_mi:]
    window_size = int(0.4 * walk_length)

    results: List[np.ndarray] = []
    for fold in range(folds):
        cache = os.path.join(out_dir, f"{fold}_saved_{walk_length}_walks.json") if out_dir else None
        if cache and os.path.exists(cache):
            walk_adj = read_json2arr(cache)
        else:
            con_us = cal_conUs(adj.shape[0], seed)
            _walks, walk_dict = sample_walks(
                adj, con_us, walk_type, p, q, walk_length, window_size)
            if out_dir:
                save_causal_walks(out_dir, walk_dict, walk_length, fold)
                walk_adj = read_json2arr(cache)
            else:
                walk_adj = _dict_to_matrix(walk_dict, adj.shape[0])
        results.append(walk_metrics(true_block, walk_adj[:n_lnc, n_lnc + n_mi:], k=k))

    matrix = np.vstack(results)
    return {
        "mean": dict(zip(WALK_METRIC_NAMES, matrix.mean(axis=0).tolist())),
        "std": dict(zip(WALK_METRIC_NAMES, matrix.std(axis=0).tolist())),
        "walk_type": walk_type,
        "task": task.name,
    }


def _dict_to_matrix(walk_dict, n_nodes: int) -> np.ndarray:
    """Same as :func:`read_json2arr` but for an in-memory walk dictionary."""
    matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for node, neighbours in walk_dict.items():
        if isinstance(neighbours[0], (list, tuple, np.ndarray)) and (
                len(neighbours) and not np.isscalar(neighbours[0])):
            neighbours = neighbours[0]
        for neighbour in neighbours:
            if int(neighbour) != int(node):
                matrix[int(node)][int(neighbour)] = 1
    return matrix
