"""Command line interface.

Replaces the ``if __name__ == "__main__"`` blocks of ``new_main.py``. The heavy
imports (DGL, PyTorch) happen inside :func:`cmd_train` so that ``--help`` and
``cw-hgt info`` work in a bare environment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Any, Dict, List

from cw_hgt import config as config_module
from cw_hgt.constants import TASKS

__all__ = ["main", "build_parser"]

#: CLI flag -> Config field, for flags whose spelling differs.
_ALIASES = {
    "task": "DataClass",
    "walk_length": "WalkLength",
    "multi_walk": "NumMultiWalk",
    "neg_samples": "num_NegativeSample",
    "hid_dim": "HidDim",
    "out_dim": "OutDim",
    "layers": "nLayers",
    "headers": "nHeaders",
    "hetero": "isHetero",
    "folds": "kFold",
    "knn": "kNN",
    "workers": "n_workers",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cw-hgt",
        description="CW-HGT: lncRNA/miRNA abiotic-stress association prediction",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run k-fold cross-validation")
    train.add_argument("--task", "--DataClass", dest="task", default="lnc_mi_stress_graphs",
                       choices=sorted(TASKS), help="which graph/task to train on")
    train.add_argument("--variant", default="default", choices=["default", "ablation"],
                       help="default builder or the ablation variant")
    train.add_argument("--pre-edge-type", default=None, choices=["l_s", "m_s"],
                       help="association edge to predict (default: task default)")
    train.add_argument("--walk-type", default="CW", choices=["CW", "RW", "N2VRW", "None"])
    train.add_argument("--walk-length", default="5", help="comma separated list of walk lengths")
    train.add_argument("--multi-walk", type=int, default=1)
    train.add_argument("--layers", default="5", help="comma separated list of HGT layer counts")
    train.add_argument("--headers", default="2", help="comma separated list of attention heads")
    train.add_argument("--hid-dim", type=int, default=128)
    train.add_argument("--out-dim", type=int, default=128)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--folds", type=int, default=10)
    train.add_argument("--knn", type=int, default=15)
    train.add_argument("--neg-samples", type=int, default=1)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--weight-decay", type=float, default=None)
    train.add_argument("--dropout", type=float, default=None)
    train.add_argument("--p", type=float, default=None, help="return parameter of the causal walk")
    train.add_argument("--q", type=int, default=None, help="in-out parameter of the causal walk")
    train.add_argument("--seed", type=int, default=None, help="base seed; fold i uses seed - i")
    train.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    train.add_argument("--data-dir", default=None, help="directory holding adj/ and sim/")
    train.add_argument("--output-dir", default=None, help="root for runs, walks and reports")
    train.add_argument("--name", default="debug", help="run name used in output paths")
    train.add_argument("--save-checkpoints", action="store_true",
                       help="write one .pth per fold (needed by the case study)")
    train.add_argument("--config", default=None, help="YAML file with Config fields")
    train.add_argument("--quiet", action="store_true", help="no progress bars")
    train.add_argument("--json-out", default=None, help="where to write the metrics report")

    info = sub.add_parser("info", help="print the effective configuration")
    info.add_argument("--config", default=None, help="YAML file with Config fields")

    return parser


def _as_list(value: Any) -> List[Any]:
    """``"5,10"`` -> ``[5, 10]`` preserving the element type of the input."""
    if not isinstance(value, str):
        return [value]
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) if p.lstrip("-").isdigit() else p for p in parts]


def _apply_config_file(cfg, path: str) -> None:
    loaded = config_module.Config.from_yaml(path)
    for key, value in loaded.as_dict().items():
        setattr(cfg, key, value)


def cmd_info(args: argparse.Namespace) -> int:
    cfg = config_module.get_config()
    if args.config:
        _apply_config_file(cfg, args.config)
    print(json.dumps(cfg.as_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    # Imported here so `cw-hgt info` works without DGL/PyTorch installed.
    from cw_hgt.data import get_dataset
    from cw_hgt.training.cv import run_cross_validation
    from cw_hgt.utils.io import write_json

    cfg = config_module.get_config()
    if args.config:
        _apply_config_file(cfg, args.config)

    # explicit CLI values win over the config file
    for dest, attr in _ALIASES.items():
        value = getattr(args, dest, None)
        if value is not None:
            setattr(cfg, attr, value)
    if args.pre_edge_type is not None:
        cfg.pre_edge_type = args.pre_edge_type
    for flag in ("epochs", "lr", "weight_decay", "dropout", "p", "q", "seed", "device",
                 "data_dir", "output_dir"):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(cfg, flag, value)
    cfg.name = args.name
    cfg.walk_type = args.walk_type
    cfg.dataset_dir = cfg.dataset_dir or "dataset"
    cfg.validate()

    print(f"[cw-hgt] task={cfg.DataClass} variant={args.variant} "
          f"edge={cfg.pre_edge_type} device={config_module.get_device()}")

    dataset = get_dataset(cfg.DataClass, variant=args.variant)
    hg = dataset[0]
    print(f"[cw-hgt] graph: {hg}")

    reports = []
    for walk_length, layers, headers in itertools.product(
            _as_list(args.walk_length), _as_list(args.layers), _as_list(args.headers)):
        cfg.WalkLength = int(walk_length)
        cfg.nLayers = int(layers)
        cfg.nHeaders = int(headers)
        cfg.validate()
        tag = f"{cfg.kNN}_{walk_length}_{layers}_{headers}"
        print(f"[cw-hgt] === walk_length={walk_length} layers={layers} heads={headers} ===")
        result = run_cross_validation(dataset, hg, cfg, verbose=not args.quiet,
                                      save_checkpoints=args.save_checkpoints)
        summary = result.summary()
        reports.append({"setting": tag, "config": cfg.as_dict(), "summary": summary,
                        "per_fold": result.metrics.tolist()})
        print(f"[cw-hgt] {tag}: " + "  ".join(
            f"{k}={v['mean']:.4f}±{v['std']:.4f}" for k, v in summary.items()))

    out = args.json_out or os.path.join(
        cfg.output_dir, cfg.DataClass, cfg.pre_edge_type, "metrics.json")
    write_json(out, reports)
    print(f"[cw-hgt] metrics written to {out}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "info":
        return cmd_info(args)
    if args.command == "train":
        return cmd_train(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
