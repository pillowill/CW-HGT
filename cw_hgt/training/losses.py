"""Training objective and negative-edge sampling."""

from __future__ import annotations

import dgl
import torch
import torch.nn.functional as F

__all__ = ["compute_loss", "construct_negative_graph"]


def compute_loss(pos_score, neg_score):
    """BCE on the concatenated scores plus the margin term, averaged.

    ``loss = (BCE(pos||neg) + mean(relu(1 - pos + neg))) / 2``
    """
    pos_labels = torch.ones_like(pos_score)
    neg_labels = torch.zeros_like(neg_score)
    labels = torch.cat([pos_labels, neg_labels])
    scores = torch.cat([pos_score, neg_score])
    loss_bce = F.binary_cross_entropy(torch.sigmoid(scores), labels)
    n_edges = pos_score.shape[0]
    loss_margin = (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()
    return (loss_bce + loss_margin) / 2


def construct_negative_graph(graph, k, etype):
    """Sample ``k`` random destination nodes per edge of ``etype``.

    Kept for completeness / for callers that need on-the-fly negative graphs.
    The main pipeline uses the negative graphs built by the dataset classes.
    """
    utype, _, vtype = etype
    src, _dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})
