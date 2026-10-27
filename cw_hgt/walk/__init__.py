"""Walk samplers used by the model.

``CausalWalk`` (this work) and the random-walk baselines live in
:mod:`cw_hgt.walk.causal`, which requires the compiled ``c_utils`` extension.
Attributes here are lazy so that importing the package does not force that
dependency.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CausalWalk", "MultiCausalWalk", "RandomWalk", "N2V_RandomWalk"]

_LAZY = {
    name: ("cw_hgt.walk.causal", name)
    for name in ("CausalWalk", "MultiCausalWalk", "RandomWalk", "N2V_RandomWalk")
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = _LAZY[name]
    return getattr(importlib.import_module(module_name), attr)


def __dir__() -> list:
    return sorted(__all__)
