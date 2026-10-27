"""Meta-path attention reduction.

``HGT`` records the attention logits of every meta-path per layer. The helpers
here collapse them into the 3x3 meta-path matrix that the paper's figures use.
"""

from __future__ import annotations

import numpy as np

__all__ = ["attn_arr", "META_PATH_ORDER"]

#: Position of each captured attention score in the 3x3 matrix (row, col).
#: Row/col 0 = lnc, 1 = mi, 2 = stress.
META_PATH_ORDER = (
    (0, 0, 0), (0, 1, 2), (0, 2, 1),
    (1, 0, 6), (1, 1, 4), (1, 2, 5),
    (2, 0, 3), (2, 1, 7), (2, 2, 8),
)


def attn_arr(meta_attn_scores, n_layers):
    """Reduce per-layer attention logits to ``{layer: 3x3 ndarray}``.

    ``meta_attn_scores[layer]`` holds one tensor per meta-path (nine for the
    three-node graph); each is reduced with ``max``. Requires exactly nine
    entries per layer, i.e. only valid for ``lnc_mi_stress_graphs``.
    """
    meta_attn_arrs = {}
    for layer in range(n_layers):
        scores = [np.max(entry.cpu().numpy()) for entry in meta_attn_scores[layer]]
        arr = np.zeros(shape=(3, 3))
        for row, col, index in META_PATH_ORDER:
            arr[row, col] = scores[index]
        meta_attn_arrs[layer] = arr
    return meta_attn_arrs
