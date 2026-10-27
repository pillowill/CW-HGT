"""Optimisers: the vendored Ranger optimiser."""

from __future__ import annotations

__all__ = ["Ranger"]


def __getattr__(name: str):
    if name != "Ranger":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from cw_hgt.optim.ranger import Ranger

    return Ranger
