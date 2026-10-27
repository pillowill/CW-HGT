"""Binary link-prediction metrics.

Column order matches :data:`cw_hgt.constants.METRIC_NAMES`:
``(auc, aupr, precision, recall, accuracy, f1, specificity)``.

The binarisation threshold is the one used in the reference scripts: the mean of
``sigmoid(score)`` over the evaluated edges. It is kept as the default because
changing it changes every reported number; pass ``threshold=0.5`` for a fixed
cut-off instead.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = ["calc_metrics", "METRIC_NAMES"]

METRIC_NAMES: Sequence[str] = (
    "auc", "aupr", "precision", "recall", "accuracy", "f1", "specificity",
)


def calc_metrics(scores, labels, threshold: Optional[float] = None) -> np.ndarray:
    """Return ``(auc, aupr, precision, recall, accuracy, f1, specificity)``.

    Parameters
    ----------
    scores
        Raw logits (positive class) for every evaluated edge; tensor or array.
    labels
        0/1 labels aligned with ``scores``; tensor or array.
    threshold
        Probability cut-off. ``None`` uses ``mean(sigmoid(scores))`` as in the
        reference implementation.

    Notes
    -----
    The confusion matrix is read as ``[[TN, FP], [FN, TP]]`` (this corrects the
    swapped indices of the retired ``main.py``; ``new_main.py`` already used this
    layout).
    """
    import torch

    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    if threshold is None:
        probs = torch.sigmoid(torch.as_tensor(scores_np)).numpy()
        preds = (probs > probs.mean()).astype(int)
    else:
        probs = torch.sigmoid(torch.as_tensor(scores_np)).numpy()
        preds = (probs > threshold).astype(int)

    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return np.array([
        roc_auc_score(labels_np, scores_np),
        average_precision_score(labels_np, scores_np),
        precision_score(labels_np, preds, zero_division=0),
        recall_score(labels_np, preds, zero_division=0),
        accuracy_score(labels_np, preds),
        f1_score(labels_np, preds, zero_division=0),
        specificity,
    ])
