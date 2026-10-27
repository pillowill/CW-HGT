"""CW-HGT: lncRNA / miRNA / abiotic-stress association prediction.

The package is intentionally importable without PyTorch or DGL: only
:mod:`cw_hgt.config`, :mod:`cw_hgt.constants` and :mod:`cw_hgt.utils` are
dependency-free. The model, data and training modules import the deep-learning
stack on demand.
"""

from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["__version__"]
