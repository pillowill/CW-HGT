"""Central configuration for CW-HGT.

The original scripts parsed the command line at *import* time from three
different modules, which made the code impossible to import as a library and
meant that any unknown CLI flag crashed the import. This module replaces that
with a single mutable configuration object:

*   :data:`_ACTIVE` holds one :class:`Config` instance, created with the
    defaults of the original ``set_params.get_args()``.
*   Legacy modules (``models.cw_gnn``, ``data.*``, ...) still do
    ``args = config.get_args()`` at import, so their bodies are unchanged, but
    they now receive this object instead of an ``argparse.Namespace``.
*   :func:`configure` / :func:`get_config` mutate that shared instance, so a
    value set by the CLI (or a test) is visible everywhere.

Keeping one shared object — rather than passing a config through every call —
is deliberate: it preserves the numerics of the reference implementation,
whose helpers consume the global NumPy RNG and read hyper-parameters lazily.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Optional

__all__ = [
    "Config",
    "get_config",
    "get_args",
    "configure",
    "reset_config",
    "get_device",
    "DEFAULTS",
]

# Walk lengths are also used to derive the window size: window = 0.4 * walk_length
WINDOW_RATIO = 0.4


@dataclasses.dataclass
class Config:
    """Hyper-parameters of one CW-HGT run.

    Field names intentionally mirror the original ``argparse`` destinations
    (``DataClass``, ``HidDim``, ...) so that existing code can keep reading
    ``args.<name>``. New fields are snake_case.
    """

    # --- dataset / task -----------------------------------------------------
    #: one of ``lnc_mi_stress_graphs``, ``lnc_stress_graphs``, ``mi_stress_graphs``
    DataClass: str = "lnc_mi_stress_graphs"
    #: which association edge is predicted: ``l_s`` or ``m_s``
    pre_edge_type: str = "l_s"
    #: node counts; overwritten from the built graph before the model is created
    num_lnc: int = 175
    num_mi: int = 500
    num_stress: int = 50
    #: number of negative edges sampled per positive edge
    num_NegativeSample: int = 1
    #: k of the kNN similarity edges
    kNN: int = 15
    #: number of cross-validation folds
    kFold: int = 10

    # --- causal walk --------------------------------------------------------
    #: ``CW`` (causal walk) or one of the baselines ``RW`` / ``N2VRW`` / ``None``
    walk_type: str = "CW"
    p: float = 4
    q: int = 3
    #: how many independent walks are sampled per node
    NumMultiWalk: int = 1
    WalkLength: int = 0
    #: worker processes of the causal-walk sampler
    n_workers: int = 6

    # --- model --------------------------------------------------------------
    HidDim: int = 128
    OutDim: int = 128
    nLayers: int = 0
    nHeaders: int = 0
    isHetero: bool = True
    dropout: float = 0.5

    # --- optimisation -------------------------------------------------------
    epochs: int = 200
    lr: float = 0.005
    weight_decay: float = 0.001
    seed: int = 3047
    random_seed: int = 123

    # --- bookkeeping --------------------------------------------------------
    #: directory holding ``adj/`` and ``sim/`` (relative paths resolve against cwd)
    dataset_dir: str = "dataset"
    #: root for checkpoints, sampled walks and metric tables
    output_dir: str = "runs"
    #: ``auto`` | ``cuda`` | ``cpu``
    device: str = "auto"
    #: free-form run name used in output paths
    name: str = "debug"

    # ------------------------------------------------------------------ utils
    @property
    def window_size(self) -> int:
        """Context window of the walk sampler (``0.4 * WalkLength``)."""
        return int(WINDOW_RATIO * self.WalkLength)

    @property
    def target_edge(self) -> tuple:
        """Heterogeneous edge type whose links are predicted."""
        if self.DataClass == "lnc_mi_stress_graphs":
            return ("lnc", "l_s", "stress") if self.pre_edge_type == "l_s" \
                else ("mi", "m_s", "stress")
        if self.DataClass == "lnc_stress_graphs":
            return ("lnc", "l_s", "stress")
        return ("mi", "m_s", "stress")

    @property
    def walk_save_path(self) -> str:
        return os.path.join(
            self.output_dir, "walks", self.DataClass, self.pre_edge_type,
            self.walk_type or "None", self.name)

    @property
    def model_save_path(self) -> str:
        return os.path.join(self.output_dir, "models", self.name)

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return dataclasses.asdict(self)

    def validate(self) -> "Config":
        """Fail fast on combinations that the original code crashed on late."""
        allowed_tasks = {"lnc_mi_stress_graphs", "lnc_stress_graphs", "mi_stress_graphs"}
        if self.DataClass not in allowed_tasks:
            raise ValueError(f"DataClass must be one of {sorted(allowed_tasks)}, "
                             f"got {self.DataClass!r}")
        if self.pre_edge_type not in {"l_s", "m_s"}:
            raise ValueError(f"pre_edge_type must be 'l_s' or 'm_s', "
                             f"got {self.pre_edge_type!r}")
        if self.walk_type not in {"CW", "RW", "N2VRW", "None", None}:
            raise ValueError(f"unknown walk_type {self.walk_type!r}")
        if self.nHeaders < 1:
            # GNNs.HGTLayer computes d_k = out_dim // n_heads
            raise ValueError("nHeaders must be >= 1 (0 raises ZeroDivisionError "
                             "in the HGT layer)")
        if self.nLayers < 1:
            raise ValueError("nLayers must be >= 1, otherwise no HGT layer is built")
        if self.kFold < 2:
            raise ValueError("kFold must be >= 2")
        if self.WalkLength < 1:
            raise ValueError("WalkLength must be >= 1")
        if self.OutDim % self.nHeaders:
            raise ValueError(f"OutDim ({self.OutDim}) must be divisible by "
                             f"nHeaders ({self.nHeaders})")
        return self

    # ------------------------------------------------------------ (de)serial.
    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyYAML is required to read YAML configs; "
                              "use from_dict()/from_json() instead") from exc
        with open(path, encoding="utf-8") as fh:
            values = yaml.safe_load(fh) or {}
        return cls.from_dict({k: v for k, v in values.items() if v is not None})

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.as_dict(), fh, sort_keys=False, allow_unicode=True)


#: Defaults, kept for tooling that wants a pristine copy.
DEFAULTS = Config()

_ACTIVE: Optional[Config] = None


def get_config() -> Config:
    """Return the process-wide configuration, creating it on first use."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = Config()
    return _ACTIVE


def get_args() -> Config:
    """Alias kept for the original modules, which call ``config.get_args()``."""
    return get_config()


def configure(**values: Any) -> Config:
    """Update the active configuration in place and return it."""
    cfg = get_config()
    for key, value in values.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"{type(cfg).__name__} has no field {key!r}")
        setattr(cfg, key, value)
    return cfg


def reset_config() -> Config:
    """Replace the active configuration with a fresh default instance."""
    global _ACTIVE
    _ACTIVE = Config()
    return _ACTIVE


def get_device():
    """Resolve ``Config.device`` to a :class:`torch.device`.

    ``auto`` prefers CUDA and falls back to CPU. Imported lazily so that
    ``cw_hgt.config`` can be used without PyTorch installed.
    """
    import torch

    cfg = get_config()
    choice = cfg.device
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)
