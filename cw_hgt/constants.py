"""Task registry: which node/edge types and data builder belong to each task.

The three graph builders in :mod:`cw_hgt.data` were previously selected by a
string literal repeated in every entry script. This module records that mapping
in one place, together with the edge type that is predicted.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

__all__ = ["Task", "TASKS", "METRIC_NAMES", "get_task"]

#: Column order of the metrics vector returned by :func:`cw_hgt.training.metrics.calc_metrics`.
METRIC_NAMES: Tuple[str, ...] = (
    "auc", "aupr", "precision", "recall", "accuracy", "f1", "specificity",
)


@dataclasses.dataclass(frozen=True)
class Task:
    """Static description of one prediction task."""

    name: str
    #: importable module that defines ``MYdataset``
    builder: str
    node_types: Tuple[str, ...]
    edge_types: Tuple[str, ...]
    #: association edge types that can be predicted, mapped to their triplet
    targets: Dict[str, Tuple[str, str, str]]

    @property
    def default_target(self) -> str:
        return "l_s" if "l_s" in self.targets else "m_s"


TASKS: Dict[str, Task] = {
    "lnc_mi_stress_graphs": Task(
        name="lnc_mi_stress_graphs",
        builder="cw_hgt.data.lnc_mi_stress",
        node_types=("lnc", "mi", "stress"),
        edge_types=("l_s", "s_l", "m_s", "s_m", "l_m", "m_l", "l_l", "m_m", "s_s"),
        targets={
            "l_s": ("lnc", "l_s", "stress"),
            "m_s": ("mi", "m_s", "stress"),
        },
    ),
    "lnc_stress_graphs": Task(
        name="lnc_stress_graphs",
        builder="cw_hgt.data.lnc_stress",
        node_types=("lnc", "stress"),
        edge_types=("l_s", "s_l", "l_l", "s_s"),
        targets={"l_s": ("lnc", "l_s", "stress")},
    ),
    "mi_stress_graphs": Task(
        name="mi_stress_graphs",
        builder="cw_hgt.data.mi_stress",
        node_types=("mi", "stress"),
        edge_types=("m_s", "s_m", "m_m", "s_s"),
        targets={"m_s": ("mi", "m_s", "stress")},
    ),
}


def get_task(name: str) -> Task:
    """Look up a task by its ``--DataClass`` name."""
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; expected one of {sorted(TASKS)}") from None
