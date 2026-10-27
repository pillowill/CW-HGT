"""Ablation variants of the graph builders.

These mirror :mod:`cw_hgt.data.lnc_mi_stress` and :mod:`cw_hgt.data.lnc_stress`
with the ablation settings (``StandardScaler`` features, symmetrised negative
graph, padded node blocks). They are kept as separate modules because merging
them behind a flag would have changed the default numbers.
"""

from __future__ import annotations

__all__ = ["lnc_mi_stress", "lnc_stress"]
