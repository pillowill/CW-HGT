"""Model definitions.

Attributes are resolved lazily so that ``import cw_hgt.models`` does not pull in
PyTorch/DGL, and so that the mutual references between the model and the encoder
modules cannot deadlock at import time.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CW_GNN",
    "ResNet",
    "DotProductPredictor",
    "HeteroDotProductPredictor",
    "HGT",
    "HGTLayer",
    "GCN",
    "GIN",
]

_LAZY = {
    "CW_GNN": ("cw_hgt.models.cw_gnn", "CW_GNN"),
    "ResNet": ("cw_hgt.models.cw_gnn", "ResNet"),
    "DotProductPredictor": ("cw_hgt.models.cw_gnn", "DotProductPredictor"),
    "HeteroDotProductPredictor": ("cw_hgt.models.cw_gnn", "HeteroDotProductPredictor"),
    "HGT": ("cw_hgt.models.encoders", "HGT"),
    "HGTLayer": ("cw_hgt.models.encoders", "HGTLayer"),
    "GCN": ("cw_hgt.models.encoders", "GCN"),
    "GIN": ("cw_hgt.models.encoders", "GIN"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = _LAZY[name]
    return getattr(importlib.import_module(module_name), attr)


def __dir__() -> list:
    return sorted(__all__)
