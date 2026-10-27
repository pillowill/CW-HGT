"""Case study: rank candidate lncRNA-stress associations (the retired ``CaseStudy.py``).

The pipeline is the one used for the paper's case-study table:

1. build a negative graph by uniform negative sampling of the ``l_s`` edges
   (every other relation keeps its observed edges);
2. take the complement of the observed adjacency as the candidate set;
3. train ``CW-GNN`` on the full graph for a few epochs;
4. score every candidate ``(lncRNA, stress)`` pair that the complement exposes;
5. rank them and optionally join the lncRNA / stress names.

Unlike the original script this is a function: paths, epochs and the output table
are arguments, and the candidate count is taken from the data instead of a
hard-coded ``reshape(8498, 1)``.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import dgl
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cw_hgt import config as config_module
from cw_hgt.models.cw_gnn import CW_GNN
from cw_hgt.optim.ranger import Ranger
from cw_hgt.training.losses import compute_loss
from cw_hgt.utils import cal_conUs, get_perturbation, set_random_seed

__all__ = ["run_case_study", "build_negative_graph", "build_candidate_graph"]

#: relations copied verbatim from the observed graph into the negative graph
_TRANSPORT_RELATIONS = ("l_l", "m_m", "s_s", "l_m", "m_l", "m_s", "s_m")


def build_negative_graph(hg) -> dgl.DGLHeteroGraph:
    """Uniformly resample the ``l_s`` edges, keep every other relation as-is."""
    neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(1, exclude_self_loops=True)
    sampled = neg_sampler(hg, {"l_s": hg.edges(etype="l_s", form="eid")})

    src_key = ("lnc", "l_s", "stress")
    neg_eids: Dict = dict(sampled)
    neg_eids[("stress", "s_l", "lnc")] = (sampled[src_key][1], sampled[src_key][0])
    for relation in _TRANSPORT_RELATIONS:
        triplet = next(t for t in hg.canonical_etypes if t[1] == relation)
        neg_eids[triplet] = hg.edge_type_subgraph([relation]).find_edges(
            hg.edges(etype=relation, form="eid"))

    neg_g = dgl.heterograph(neg_eids)
    neg_g.ndata["h"] = hg.ndata["h"]
    return neg_g


def build_candidate_graph(candidate_adj: np.ndarray, node_feats: Dict[str, torch.Tensor],
                          k: int) -> dgl.DGLHeteroGraph:
    """Heterogeneous graph over the candidate pairs plus kNN similarity edges."""
    n_lnc = node_feats["lnc"].shape[0]
    n_mi = node_feats["mi"].shape[0]

    knn = {name: dgl.knn_graph(feats, k=k, dist="cosine") for name, feats in node_feats.items()}
    ll_src, ll_dst = knn["lnc"].edges()
    mm_src, mm_dst = knn["mi"].edges()
    ss_src, ss_dst = knn["stress"].edges()

    l_s_row, l_s_col = np.nonzero(candidate_adj[:n_lnc, n_lnc + n_mi:])
    m_s_row, m_s_col = np.nonzero(candidate_adj[n_lnc:n_lnc + n_mi, n_lnc + n_mi:])
    l_m_row, l_m_col = np.nonzero(candidate_adj[:n_lnc, n_lnc:n_lnc + n_mi])

    def t(arr):
        return torch.as_tensor(arr, dtype=torch.int64)

    graph = dgl.heterograph({
        ("lnc", "l_s", "stress"): (t(l_s_row), t(l_s_col)),
        ("stress", "s_l", "lnc"): (t(l_s_col), t(l_s_row)),
        ("mi", "m_s", "stress"): (t(m_s_row), t(m_s_col)),
        ("stress", "s_m", "mi"): (t(m_s_col), t(m_s_row)),
        ("lnc", "l_m", "mi"): (t(l_m_row), t(l_m_col)),
        ("mi", "m_l", "lnc"): (t(l_m_col), t(l_m_row)),
        ("lnc", "l_l", "lnc"): (ll_src, ll_dst),
        ("mi", "m_m", "mi"): (mm_src, mm_dst),
        ("stress", "s_s", "stress"): (ss_src, ss_dst),
    })
    graph.ndata["h"] = node_feats
    return graph


def run_case_study(dataset, hg, cfg=None, *, epochs: Optional[int] = None,
                   device=None, out_csv: Optional[str] = None,
                   map_ids: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Train on the full graph and rank the candidate lncRNA-stress pairs.

    Returns a :class:`pandas.DataFrame` with columns
    ``lncRNA_id``, ``stress_id``, ``score`` (plus ``lncRNA_name`` /
    ``stress_name`` when ``map_ids`` is true and the id tables exist).
    """
    cfg = cfg or config_module.get_config()
    if cfg.DataClass != "lnc_mi_stress_graphs":
        raise ValueError("the case study is defined for lnc_mi_stress_graphs only")
    device = device or config_module.get_device()
    epochs = cfg.epochs if epochs is None else epochs
    set_random_seed(cfg.seed)

    node_feats = hg.ndata["h"]
    n_lnc, n_mi = hg.num_nodes("lnc"), hg.num_nodes("mi")
    all_adj = dataset.get_adj(hg)
    case_adj = np.ones_like(all_adj) - all_adj

    neg_g = build_negative_graph(hg)
    all_neg_adj = dataset.get_adj(neg_g)
    case_hg = build_candidate_graph(case_adj, node_feats, cfg.kNN)

    node_dict = {ntype: idx for idx, ntype in enumerate(hg.ntypes)}
    edge_dict = {etype: idx for idx, etype in enumerate(hg.etypes)}
    in_dims = [feats.shape[1] for feats in node_feats.values()]

    con_us = cal_conUs(all_adj.shape[0], cfg.seed)
    blocks, start = {}, 0
    for name, feats in node_feats.items():
        blocks[name] = con_us[start:start + feats.shape[0]]
        start += feats.shape[0]
    perturbed = {name: torch.from_numpy(get_perturbation(blocks[name], feats))
                 for name, feats in node_feats.items()}

    model = CW_GNN(in_dims, cfg.HidDim, cfg.OutDim, all_adj, con_us, cfg.p, cfg.q,
                   cfg.WalkLength, cfg.window_size, node_dict, edge_dict,
                   n_layers=cfg.nLayers, n_headers=cfg.nHeaders,
                   multi_n=cfg.NumMultiWalk, is_hetero=cfg.isHetero).to(device)
    optimiser = Ranger(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pbar = tqdm(total=epochs, disable=not verbose, desc="case-study fit", ncols=80)
    for _ in range(epochs):
        model.train()
        pos_score, neg_score, _walks = model(
            all_adj, all_neg_adj, perturbed, hg.to(device), neg_g.to(device),
            cfg.target_edge)
        loss = compute_loss(pos_score, neg_score)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        pbar.update(1)
        pbar.set_postfix(loss=f"{float(loss):.4f}")
    pbar.close()

    model.eval()
    with torch.no_grad():
        scores, _, _ = model(case_adj, case_adj, perturbed, case_hg, case_hg,
                             cfg.target_edge)

    l_s_row, l_s_col = np.nonzero(case_adj[:n_lnc, n_lnc + n_mi:])
    ranked = pd.DataFrame({
        "lncRNA_id": l_s_row,
        "stress_id": l_s_col,
        "score": torch.sigmoid(scores).detach().cpu().numpy().reshape(-1),
    }).sort_values("score", ascending=False).reset_index(drop=True)

    if map_ids:
        ranked = _map_names(ranked, cfg.dataset_dir)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        ranked.to_csv(out_csv, index=False)
    return ranked


def _map_names(ranked: pd.DataFrame, dataset_dir: str) -> pd.DataFrame:
    """Join lncRNA/stress display names when the id tables are available."""
    lnc_path = os.path.join(dataset_dir, "lnc_ids.csv")
    stress_path = os.path.join(dataset_dir, "stress_ids.csv")
    if not (os.path.exists(lnc_path) and os.path.exists(stress_path)):
        return ranked
    lnc_id = pd.read_csv(lnc_path, header=0, names=["lncRNA_name", "lncRNA_id"])
    stress_id = pd.read_csv(stress_path, header=0, names=["stress_name", "stress_id"])
    merged = ranked.merge(lnc_id, on="lncRNA_id", how="left")
    merged = merged.merge(stress_id, on="stress_id", how="left")
    return merged[["lncRNA_name", "stress_name", "score", "lncRNA_id", "stress_id"]]
