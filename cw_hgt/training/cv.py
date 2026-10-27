"""K-fold training / evaluation loop.

This is the consolidated version of the original ``new_main.py`` (the entry
point the paper's numbers came from). The control flow, call order and RNG usage
are preserved; what changed is that nothing is read from module-level globals any
more, so the function can be imported, tested and called with any configuration.

Per fold:

1. seed with ``cfg.seed - fold`` and draw the conditioning vector ``Con_Us``
   with the same seed (as the original ``cal_conUs(num_nodes, seed=3047 - i)``);
2. build the perturbed node features with :func:`cw_hgt.utils.get_perturbation`;
3. train ``CW-GNN`` on the perturbed train graph with Ranger;
4. evaluate on the held-out edges using the *unperturbed* features, which is what
   ``new_main.py`` did (the perturbation is a training-time intervention).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from cw_hgt import config as config_module
from cw_hgt.models import cw_gnn
from cw_hgt.models.cw_gnn import CW_GNN
from cw_hgt.optim.ranger import Ranger
from cw_hgt.training.losses import compute_loss
from cw_hgt.training.metrics import METRIC_NAMES, calc_metrics
from cw_hgt.utils import cal_conUs, get_perturbation, save_causal_walks, set_random_seed

__all__ = ["CVResult", "run_cross_validation", "node_dicts"]


@dataclasses.dataclass
class CVResult:
    """Outcome of one k-fold run."""

    #: ``(k_fold, 7)`` metric matrix, columns in :data:`METRIC_NAMES`
    metrics: np.ndarray
    #: mean / std over folds
    mean: np.ndarray
    std: np.ndarray
    #: per-fold ``{"scores": ndarray, "labels": ndarray}`` for ROC/PR curves
    folds: List[Dict[str, np.ndarray]]
    #: per-fold training loss trace
    history: List[List[float]]
    #: trained models (``None`` when ``keep_models=False``)
    models: Optional[List[torch.nn.Module]] = None

    def summary(self) -> Dict[str, Dict[str, float]]:
        """``{metric: {"mean": ..., "std": ...}}`` plus one row per fold."""
        return {
            name: {"mean": float(mean), "std": float(std)}
            for name, mean, std in zip(METRIC_NAMES, self.mean, self.std)
        }


def node_dicts(hg) -> tuple:
    """``(node_dict, edge_dict)``: type name -> index, in graph order."""
    node_dict = {ntype: idx for idx, ntype in enumerate(hg.ntypes)}
    edge_dict = {etype: idx for idx, etype in enumerate(hg.etypes)}
    return node_dict, edge_dict


def _split_conditioning(con_us: np.ndarray, node_feats: Dict[str, Any]) -> Dict[str, Any]:
    """Slice the flat conditioning vector into one block per node type.

    Mirrors the per-task slicing of ``new_main.py`` but derives the block sizes
    from the feature dictionary itself, so it cannot drift from the graph.
    """
    blocks, start = {}, 0
    for ntype, feats in node_feats.items():
        stop = start + feats.shape[0]
        blocks[ntype] = con_us[start:stop]
        start = stop
    if start != con_us.shape[0]:
        raise ValueError(f"conditioning vector has {con_us.shape[0]} entries but the "
                         f"graph has {start} nodes")
    return blocks


def run_cross_validation(dataset, hg, cfg=None, *, node_feats: Optional[Dict] = None,
                         verbose: bool = True, keep_models: bool = False,
                         save_checkpoints: bool = False) -> CVResult:
    """Train and evaluate ``CW-GNN`` with k-fold cross-validation.

    Parameters
    ----------
    dataset
        A builder instance from :mod:`cw_hgt.data` (provides
        ``split_train_test``, ``negative_sample`` and ``get_adj``).
    hg
        The heterogeneous graph, i.e. ``dataset[0]``.
    cfg
        Configuration object; defaults to the process-wide one.
    node_feats
        Node features; defaults to ``hg.ndata["h"]``.
    save_checkpoints
        Write ``<model_save_path>/<...>_<fold>.pth`` per fold, like the retired
        ``main.py`` did (used by the case study).
    """
    cfg = cfg or config_module.get_config()
    cfg.validate()
    device = config_module.get_device()
    # The model's forward passes read its module-level `device`; keep it in sync
    # with the device resolved here so `--device cpu` cannot mix devices.
    cw_gnn.set_device(device)
    node_feats = hg.ndata["h"] if node_feats is None else node_feats

    # The model slices the block adjacency with these counts; deriving them from
    # the graph removes the drift between --num_lnc and the actual node count.
    for ntype in ("lnc", "mi", "stress"):
        if ntype in hg.ntypes:
            setattr(cfg, f"num_{ntype}", hg.num_nodes(ntype))
    num_lnc = getattr(cfg, "num_lnc")
    num_mi = getattr(cfg, "num_mi")
    num_stress = getattr(cfg, "num_stress")
    if "lnc" not in hg.ntypes:
        num_lnc = 0
    if "mi" not in hg.ntypes:
        num_mi = 0
    if "stress" not in hg.ntypes:
        num_stress = 0

    node_dict, edge_dict = node_dicts(hg)
    in_dims = [feats.shape[1] for feats in node_feats.values()]
    walk_length = cfg.WalkLength
    window_size = cfg.window_size

    pos_tr, pos_te = dataset.split_train_test(hg, cfg.kFold, cfg.pre_edge_type)
    neg_tr, neg_te = dataset.negative_sample(
        hg, k=cfg.num_NegativeSample, kFold=cfg.kFold, pre_edge=cfg.pre_edge_type)

    metrics, folds, history, models = [], [], [], []
    walk_dir = os.path.join(cfg.walk_save_path, "train")

    for fold in range(cfg.kFold):
        seed = cfg.seed - fold
        set_random_seed(seed)

        tr_hg, tr_neg = pos_tr[fold], neg_tr[fold]
        te_hg, te_neg = pos_te[fold], neg_te[fold]
        adj = dataset.get_adj(tr_hg)
        neg_adj = dataset.get_adj(tr_neg)

        con_us = cal_conUs(adj.shape[0], seed)
        con_us_dict = _split_conditioning(con_us, node_feats)
        perturbed = {name: torch.from_numpy(get_perturbation(con_us_dict[name], feats))
                     for name, feats in node_feats.items()}

        model = CW_GNN(in_dims, cfg.HidDim, cfg.OutDim, adj, con_us, cfg.p, cfg.q,
                       walk_length, window_size, node_dict, edge_dict,
                       n_layers=cfg.nLayers, n_headers=cfg.nHeaders,
                       multi_n=cfg.NumMultiWalk, is_hetero=cfg.isHetero).to(device)

        walks = model.return_walk_dict()
        if walks is not None:
            save_causal_walks(walk_dir, walks, walk_length, fold)

        optimiser = Ranger(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        pbar = tqdm(total=cfg.epochs, disable=not verbose, desc=f"fold {fold}", ncols=80)
        losses = []
        for _epoch in range(cfg.epochs):
            model.train()
            pos_score, neg_score, _walks = model(
                adj, neg_adj, perturbed, tr_hg, tr_neg, cfg.target_edge)
            loss = compute_loss(pos_score, neg_score)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
            pbar.update(1)
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
        pbar.close()
        history.append(losses)

        model.eval()
        with torch.no_grad():
            te_pos, te_neg, _walks = model(
                adj, neg_adj, node_feats, te_hg, te_neg, cfg.target_edge)
        labels = torch.cat([torch.ones_like(te_pos), torch.zeros_like(te_neg)])
        scores = torch.cat([te_pos, te_neg])
        metrics.append(calc_metrics(scores, labels))
        folds.append({"scores": scores.detach().cpu().numpy(),
                      "labels": labels.detach().cpu().numpy()})

        if save_checkpoints:
            os.makedirs(cfg.model_save_path, exist_ok=True)
            stem = f"{cfg.kNN}_{cfg.HidDim}_{cfg.OutDim}_{walk_length}_{cfg.nLayers}_{cfg.nHeaders}"
            ckpt = os.path.join(cfg.model_save_path, f"{stem}_{fold}.pth")
            torch.save(model.state_dict(), ckpt)

        if keep_models:
            models.append(model)
        else:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_matrix = np.vstack(metrics)
    return CVResult(
        metrics=metric_matrix,
        mean=metric_matrix.mean(axis=0),
        std=metric_matrix.std(axis=0),
        folds=folds,
        history=history,
        models=models or None,
    )
