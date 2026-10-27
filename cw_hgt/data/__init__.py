"""Heterogeneous graph builders.

Each task has its own module exposing a ``MYdataset`` (a ``dgl`` ``DGLDataset``
subclass). :func:`get_dataset` maps a ``--DataClass`` value to the right builder.

The ``ablation`` subpackage mirrors two of the builders with the ablation
settings used in the paper (standardised features, symmetrised negative graph).
Pass ``variant="ablation"`` to :func:`get_dataset` to use them.
"""

from __future__ import annotations

import importlib
from typing import Any

from cw_hgt.constants import TASKS, get_task

__all__ = ["get_dataset", "get_builder", "TASKS"]


def get_builder(task: str, variant: str = "default"):
    """Import and return the ``MYdataset`` class for ``task``.

    Parameters
    ----------
    task
        One of ``lnc_mi_stress_graphs``, ``lnc_stress_graphs``,
        ``mi_stress_graphs``.
    variant
        ``default`` for the reported configuration, ``ablation`` for the
        ablation variant (only available for the two lnc tasks).
    """
    spec = get_task(task)
    if variant == "default":
        module_name = spec.builder
    elif variant == "ablation":
        if task not in {"lnc_mi_stress_graphs", "lnc_stress_graphs"}:
            raise ValueError(f"no ablation variant for {task!r}")
        module_name = f"cw_hgt.data.ablation.{spec.builder.rsplit('.', 1)[-1]}"
    else:
        raise ValueError(f"unknown variant {variant!r}; use 'default' or 'ablation'")
    module = importlib.import_module(module_name)
    return module.MYdataset


def get_dataset(task: str, variant: str = "default") -> Any:
    """Instantiate the dataset builder for ``task``.

    Building the graph reads ``dataset/adj`` and ``dataset/sim`` relative to the
    current working directory (see :meth:`cw_hgt.config.Config.dataset_dir`).
    """
    return get_builder(task, variant)()
