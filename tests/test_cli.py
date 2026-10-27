"""CLI surface: parser wiring and the dependency-free `info` command."""

from __future__ import annotations

import json

import pytest

from cw_hgt import config
from cw_hgt.cli import build_parser, main


def test_parser_has_expected_subcommands():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])          # a subcommand is required


def test_train_defaults():
    args = build_parser().parse_args(["train"])
    assert args.task == "lnc_mi_stress_graphs"
    assert args.walk_type == "CW"
    assert args.walk_length == "5"
    assert args.layers == "5"
    assert args.headers == "2"
    assert args.variant == "default"


def test_info_prints_defaults_as_json(capsys):
    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["DataClass"] == "lnc_mi_stress_graphs"
    assert payload["epochs"] == 200


def test_info_applies_config_file(tmp_dir):
    yaml = pytest.importorskip("yaml")
    path = tmp_dir / "cfg.yaml"
    path.write_text(yaml.safe_dump({"epochs": 11, "nLayers": 3, "nHeaders": 1,
                                    "WalkLength": 5}), encoding="utf-8")
    assert main(["info", "--config", str(path)]) == 0


def test_train_is_importable_without_dgl():
    """`cw-hgt train` imports DGL lazily, so the parser must not need it."""
    args = build_parser().parse_args(["train", "--task", "mi_stress_graphs"])
    assert args.task == "mi_stress_graphs"


def test_walk_length_list_parsing():
    from cw_hgt.cli import _as_list

    assert _as_list("5,10,15") == [5, 10, 15]
    assert _as_list("5") == [5]
    assert _as_list(7) == [7]
