"""RNG helpers, IO helpers and the attention reducer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cw_hgt.utils import (
    NumpyEncoder,
    attn_arr,
    cal_conUs,
    get_perturbation,
    save_causal_walks,
    write_json,
)


def test_cal_conUs_is_deterministic_and_in_range():
    first = cal_conUs(50, seed=3047)
    second = cal_conUs(50, seed=3047)
    assert np.array_equal(first, second)
    assert first.shape == (50,)
    assert set(np.unique(first)).issubset({1, 2})


def test_cal_conUs_depends_on_seed():
    assert not np.array_equal(cal_conUs(50, 1), cal_conUs(50, 2))


def test_cal_conUs_seeds_global_rng():
    """The reference implementation seeds NumPy as a side effect; keep that."""
    cal_conUs(10, seed=123)
    after_first = np.random.rand()
    cal_conUs(10, seed=123)
    assert np.random.rand() == pytest.approx(after_first)


def test_get_perturbation_keeps_untouched_rows():
    X = np.array([[1, 0, 1, 0, 1],
                  [0, 1, 1, 1, 0]], dtype=float)
    con_us = np.array([2, 2])
    np.random.seed(0)
    out = get_perturbation(con_us, X)
    assert np.array_equal(out, X)


def test_get_perturbation_shape_and_mask_rows():
    X = np.random.randint(0, 2, size=(40, 9)).astype(float)
    con_us = np.array([1] * 20 + [2] * 20)
    np.random.seed(7)
    out = get_perturbation(con_us, X)
    assert out.shape == X.shape
    # rows with con_us == 2 pass through untouched
    assert np.array_equal(out[20:], X[20:])
    # rows with con_us == 1 stay binary (they are kept or flipped, never rescaled)
    assert set(np.unique(out[:20])).issubset({0.0, 1.0})


def test_numpy_encoder_serialises_numpy_types():
    payload = {"int": np.int64(3), "float": np.float32(0.5), "arr": np.arange(3)}
    text = json.dumps(payload, cls=NumpyEncoder)
    assert json.loads(text) == {"int": 3, "float": 0.5, "arr": [0, 1, 2]}


def test_save_causal_walks_creates_directory(tmp_dir):
    target = tmp_dir / "nested" / "walks"
    path = save_causal_walks(str(target), {0: [1, 2], 1: [0]}, walk_len=5, idx=3)
    assert path.endswith("3_saved_5_walks.json")
    assert json.load(open(path, encoding="utf-8"))["0"] == [1, 2]


def test_write_json_creates_parents(tmp_dir):
    path = write_json(str(tmp_dir / "a" / "b.json"), {"k": np.int32(1)})
    assert json.load(open(path, encoding="utf-8")) == {"k": 1}


def test_attn_arr_layout():
    torch = pytest.importorskip("torch")
    # nine meta-paths per layer, each a 1-D tensor; layer 0 mirrors 0..8
    layer = [torch.tensor([float(i)]) for i in range(9)]
    result = attn_arr([layer], n_layers=1)
    assert set(result) == {0}
    expected = np.array([[0, 2, 1],
                         [6, 4, 5],
                         [3, 7, 8]], dtype=float)
    assert np.array_equal(result[0], expected)
