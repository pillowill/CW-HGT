"""Configuration behaviour: defaults, validation, task mapping."""

from __future__ import annotations

import pytest

from cw_hgt import config
from cw_hgt.constants import METRIC_NAMES, TASKS, get_task


def test_defaults_match_original_arguments():
    cfg = config.Config()
    assert cfg.DataClass == "lnc_mi_stress_graphs"
    assert cfg.pre_edge_type == "l_s"
    assert cfg.WalkLength == 0          # filled in by the CLI / config file
    assert cfg.epochs == 200
    assert cfg.seed == 3047
    assert cfg.HidDim == 128 and cfg.OutDim == 128
    assert cfg.dropout == 0.5
    assert cfg.isHetero is True


def test_singleton_is_shared():
    first = config.get_config()
    config.configure(epochs=3, nLayers=5, nHeaders=2)
    assert config.get_config() is first
    assert config.get_args().epochs == 3
    assert first.nLayers == 5


def test_configure_rejects_unknown_field():
    with pytest.raises(AttributeError):
        config.configure(not_a_field=1)


def test_reset_config_returns_defaults():
    config.configure(epochs=1)
    fresh = config.reset_config()
    assert fresh.epochs == 200


@pytest.mark.parametrize("edge_type,expected", [
    ("l_s", ("lnc", "l_s", "stress")),
    ("m_s", ("mi", "m_s", "stress")),
])
def test_target_edge_three_type_task(edge_type, expected):
    cfg = config.Config(pre_edge_type=edge_type)
    assert cfg.target_edge == expected


def test_target_edge_two_type_tasks():
    assert config.Config(DataClass="lnc_stress_graphs").target_edge == \
        ("lnc", "l_s", "stress")
    assert config.Config(DataClass="mi_stress_graphs").target_edge == \
        ("mi", "m_s", "stress")


def test_window_size_is_fraction_of_walk_length():
    cfg = config.Config(WalkLength=10)
    assert cfg.window_size == 4
    assert config.Config(WalkLength=5).window_size == 2


@pytest.mark.parametrize("overrides", [
    {"nHeaders": 0},
    {"nLayers": 0},
    {"kFold": 1},
    {"WalkLength": 0},
    {"DataClass": "nope"},
    {"pre_edge_type": "x"},
    {"OutDim": 127, "nHeaders": 2},   # not divisible
])
def test_validate_rejects_bad_combinations(overrides):
    with pytest.raises(ValueError):
        config.Config(**overrides).validate()


def test_validate_accepts_defaults_with_walk_length():
    config.Config(WalkLength=5, nLayers=5, nHeaders=2).validate()


def test_task_registry_is_consistent():
    assert set(TASKS) == {"lnc_mi_stress_graphs", "lnc_stress_graphs", "mi_stress_graphs"}
    for name, task in TASKS.items():
        assert task.name == name
        assert task.builder.startswith("cw_hgt.data.")
        for edge in task.targets:
            assert edge in task.edge_types
    assert get_task("mi_stress_graphs").default_target == "m_s"
    with pytest.raises(KeyError):
        get_task("unknown")


def test_metric_names_order():
    assert METRIC_NAMES == ("auc", "aupr", "precision", "recall", "accuracy",
                            "f1", "specificity")


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        config.Config.from_dict({"epochs": 5, "bogus": 1})


def test_json_roundtrip(tmp_dir):
    cfg = config.Config(epochs=7, nLayers=2, nHeaders=2, WalkLength=5, name="t")
    path = tmp_dir / "cfg.json"
    import json

    path.write_text(json.dumps(cfg.as_dict()), encoding="utf-8")
    assert config.Config.from_json(str(path)).epochs == 7


def test_yaml_roundtrip(tmp_dir):
    yaml = pytest.importorskip("yaml")
    cfg = config.Config(epochs=7, nLayers=2, nHeaders=2, WalkLength=5)
    path = tmp_dir / "cfg.yaml"
    cfg.to_yaml(str(path))
    assert config.Config.from_yaml(str(path)).nLayers == 2
