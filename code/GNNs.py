# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/23 13:12
@Auth ： He Yu
@File ：GNNs.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import numpy as np
from dgl.nn.pytorch import GraphConv, HeteroGraphConv, HGTConv, GINConv
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from set_params import get_args

args = get_args()


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats * 2)
        self.conv2 = GraphConv(h_feats * 2, h_feats * 4)
        self.conv3 = GraphConv(h_feats * 4, h_feats)
        self.conv4 = GraphConv(h_feats, num_classes)
        self.drop = nn.Dropout(dropout)
        self.reset_params()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.drop(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.drop(h)
        h = self.conv3(g, h)
        h = F.leaky_relu(h)
        h = self.drop(h)
        h = self.conv4(g, h)
        return h

    def reset_params(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()


class GIN(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, dropout):
        super(GIN, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.hidden1_dim)
        self.gin1 = GINConv(nn.Linear(self.in_dim, self.hidden1_dim), aggregator_type="mean", activation=F.leaky_relu_)
        self.gin2 = GINConv(nn.Linear(self.hidden1_dim, self.hidden2_dim), aggregator_type="sum")

    def forward(self, g, features):
        h = self.gin1(g, features)
        h = self.bn(h)
        h = self.drop(h)
        h = self.gin2(g, h)
        return h


class HGTLayer(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            node_dict,
            edge_dict,
            n_heads,
            dropout=args.dropout,
            use_norm=False,
    ):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None
        self.att_edge_score = []
        self.msg_edge_score = {}

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.n_heads)
        )
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)
        self.reset_params()

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict

            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (
                        sub_graph.edata.pop("t").sum(-1)
                        * relation_pri
                        / self.sqrt_dk
                )
                # self.att_edge_score.append(attn_score)
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")
                # self.msg_edge_score[etype] = v.clone()
                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"),
                        fn.sum("m", "t"),
                    )
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:
                """
                Step 3: Target-specific Aggregation
                x = norm( W[node_type] * gelu( Agg(x) ) + x )
                """
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h, self.att_edge_score, self.msg_edge_score

    def reset_params(self):
        for t in range(self.num_types):
            self.k_linears[t].reset_parameters()
            self.q_linears[t].reset_parameters()
            self.v_linears[t].reset_parameters()
            self.a_linears[t].reset_parameters()


class HGT(nn.Module):
    def __init__(
            self,
            node_dict,
            edge_dict,
            n_inp,
            n_hid,
            n_out,
            n_layers,
            n_heads,
            use_norm=True,
    ):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        self.n_layers_att = []
        self.n_layers_msg = []
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(
                HGTLayer(
                    n_hid,
                    n_hid,
                    node_dict,
                    edge_dict,
                    n_heads,
                    use_norm=use_norm,
                )
            )
        self.out = nn.Linear(n_hid, n_out)
        self.reset_params()

    def forward(self, G):
        h = {}

        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data["h"]))
        for i in range(self.n_layers):
            h, att_layer, msg_layer = self.gcs[i](G, h)
            self.n_layers_att.append(att_layer)
            self.n_layers_msg.append(msg_layer)

        for k, v in h.items():
            h[k] = self.out(h[k])
        return h

    def reset_params(self):
        for i in range(len(self.node_dict)):
            self.adapt_ws[i].reset_parameters()
        self.out.reset_parameters()


class dglHGT(nn.Module):
    def __init__(
            self, in_size, head_size, num_heads, num_ntypes, num_etypes, num_layers, dropout=0.2
    ):
        super(dglHGT, self).__init__()
        self.num_layers = num_layers

        # Define a stack of HGT layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(HGTConv(in_size, head_size, num_heads, num_ntypes, num_etypes, dropout))
        self.bns.append(nn.BatchNorm1d(head_size * num_heads))
        for _ in range(num_layers):
            self.layers.append(HGTConv(head_size * num_heads, head_size, num_heads, num_ntypes, num_etypes, dropout))
            self.bns.append(nn.BatchNorm1d(head_size * num_heads))
        # Final linear layer to project the output features (optional)
        self.output_proj = nn.Linear(head_size * num_heads, in_size)
        self.reset_params()

    def forward(self, g, h, ntype, etype):
        """
        Forward pass through multiple HGT layers.

        Parameters:
        g : DGLGraph
            The input graph.
        h : torch.Tensor
            The input node features.
        ntype : torch.Tensor
            Node type tensor.
        etype : torch.Tensor
            Edge type tensor.

        Returns:
        torch.Tensor
            Final node embeddings after all HGT layers.
        """
        for bn, layer in zip(self.bns, self.layers):
            h = layer(g, h, ntype, etype)  # Pass through each HGT layer
            h = F.leaky_relu(h)
            h = bn(h)


        h = self.output_proj(h)  # Project the final node embeddings (optional)
        return h

    def reset_params(self):
        self.output_proj.reset_parameters()