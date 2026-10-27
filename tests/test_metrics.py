"""Metric computation."""

from __future__ import annotations

import numpy as np
import pytest

from cw_hgt.training.metrics import METRIC_NAMES, calc_metrics

torch = pytest.importorskip("torch")


def _scores_labels(perfect=True):
    scores = torch.tensor([2.0, 1.5, -1.0, -2.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0]) if perfect else \
        torch.tensor([1.0, 0.0, 1.0, 0.0])
    return scores, labels


def test_metric_vector_shape_and_order():
    scores, labels = _scores_labels()
    result = calc_metrics(scores, labels)
    assert result.shape == (len(METRIC_NAMES),)
    assert len(METRIC_NAMES) == 7


def test_perfect_separation():
    result = calc_metrics(*_scores_labels(perfect=True))
    values = dict(zip(METRIC_NAMES, result))
    assert values["auc"] == pytest.approx(1.0)
    assert values["aupr"] == pytest.approx(1.0)
    assert values["accuracy"] == pytest.approx(1.0)
    assert values["f1"] == pytest.approx(1.0)
    assert values["specificity"] == pytest.approx(1.0)


def test_specificity_uses_tn_fp_of_correct_quadrant():
    """With swapped cm indices (the retired main.py bug) specificity would be 1.0 here."""
    scores = torch.tensor([2.0, 1.0, 0.5, -1.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    values = dict(zip(METRIC_NAMES, calc_metrics(scores, labels)))
    # threshold = mean(sigmoid) splits the two highest from the two lowest,
    # so TN=2, FP=0, FN=0, TP=2
    assert values["specificity"] == pytest.approx(1.0)
    assert values["recall"] == pytest.approx(1.0)
    assert values["precision"] == pytest.approx(1.0)


def test_fixed_threshold_changes_predictions():
    # sigmoid -> [0.953, 0.622, 0.475]; the mean cut-off (0.683) rejects the
    # middle edge, a fixed 0.5 cut-off accepts it.
    scores = torch.tensor([3.0, 0.5, -0.1])
    labels = torch.tensor([1.0, 1.0, 0.0])
    default = dict(zip(METRIC_NAMES, calc_metrics(scores, labels)))
    strict = dict(zip(METRIC_NAMES, calc_metrics(scores, labels, threshold=0.5)))
    assert default["recall"] == pytest.approx(0.5)
    assert strict["recall"] == pytest.approx(1.0)
    assert default["specificity"] == pytest.approx(1.0)
    assert strict["specificity"] == pytest.approx(1.0)


def test_accepts_numpy_inputs():
    scores = np.array([2.0, -2.0])
    labels = np.array([1, 0])
    assert calc_metrics(scores, labels)[0] == pytest.approx(1.0)
