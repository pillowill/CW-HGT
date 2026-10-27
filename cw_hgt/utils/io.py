"""JSON/NP helpers for the sampled walks and the metric tables."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import numpy as np

__all__ = ["NumpyEncoder", "ensure_dir", "save_causal_walks", "write_json", "read_json"]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars and arrays."""

    def default(self, obj):  # noqa: D102 - see json.JSONEncoder
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if needed and return it."""
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def save_causal_walks(savePath: str, walk_dict: Mapping[Any, Any], walk_len: int,
                      idx: int) -> str:
    """Dump the sampled walks of one fold to ``<savePath>/<idx>_saved_<len>_walks.json``.

    Returns the written path. The directory is created if missing (the original
    implementation raised ``FileNotFoundError`` here).
    """
    ensure_dir(savePath)
    path = os.path.join(savePath, f"{idx}_saved_{walk_len}_walks.json")
    with open(path, "w+", encoding="utf-8") as fh:
        fh.write(json.dumps(walk_dict, ensure_ascii=False, indent=4, cls=NumpyEncoder))
    return path


def write_json(path: str, payload: Any) -> str:
    """Write ``payload`` as UTF-8 JSON, creating parent directories."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    return path


def read_json(path: str) -> Any:
    """Read a UTF-8 JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
