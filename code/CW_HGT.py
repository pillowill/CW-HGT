# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/15 22:10
@Auth ： He Yu
@File ：CW_HGT.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import dgl
import numpy as np
import torch
from dgl.nn.pytorch import GraphConv, HGTConv
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from CausalWalk import MultiCausalWalk, N2V_RandomWalk, RandomWalk
import GNNs
import set_params

device = torch.device('cuda')
"""
每次epoch都需要重新采样
每次重新采样后，构建新的图，GCN在此图上进行卷积
[[S_lnc_lnc, A_lnc_mi, A_lnc_stress],
 [A_lnc_mi.T, S_mi_mi, A_mi_stress.T],
 [A_lnc_stress.T, A_mi_stress, S_stress_stress]]
"""
args = set_params.get_args()


def GNN_Hetero_G(ls, ms, lm, ll, mm, ss):
    hetero_g = dgl.heterograph({
        ('lnc', 'l_s', 'stress'): (
            ls[0], ls[1]
        ),
        ('stress', 's_l', 'lnc'): (
            ls[1], ls[0]
        ),
        ('mi', 'm_s', 'stress'): (
            ms[0], ms[1]
        ),
        ('stress', 's_m', 'mi'): (
            ms[1], ms[0]
        ),
        ('lnc', 'l_m', 'mi'): (
            lm[0], lm[1]
        ),
        ('mi', 'm_l', 'lnc'): (
            lm[1], lm[0]
        ),
        ('lnc', 'l_l', 'lnc'): (
            ll
        ),
        ('mi', 'm_m', 'mi'): (
            mm
        ),
        ('stress', 's_s', 'stress'): (
            ss
        )
    })
    return hetero_g


class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResNet, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.ac = nn.LeakyReLU()
        # 使用残差连接
        self.shortcut = nn.Linear(in_dim, out_dim)
        self.reset_params()

    def forward(self, in_feat):
        x = self.linear(in_feat)
        x = self.bn(x)
        x = self.ac(x)
        res = self.shortcut(in_feat)
        out = torch.add(x, res)
        return out

    def reset_params(self):
        self.linear.reset_parameters()
        self.shortcut.reset_parameters()
        self.bn.reset_parameters()


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对异构图的每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class CW_GNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, adj, Con_Us, p, q, walk_length, window_size,
                 node_dict, edge_dict, is_hetero, multi_n=1):
        super(CW_GNN, self).__init__()
        n_layers = args.nLayers
        n_headers = args.nHeaders
        self.adj = adj
        self.Con_Us = Con_Us
        n_nodes = self.adj.shape[0]
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.is_hetero = is_hetero
        self.multi_n = multi_n
        self.n_lnc, self.n_mi, self.n_stress = args.num_lnc, args.num_mi, args.num_stress
        self.cus_adj = np.zeros_like(self.adj)
        # 基于CW
        if args.walk_type == 'CW':
            self.causal_walker = MultiCausalWalk(p, q, walk_length, window_size, multi_n)
            self.causal_walks_list, self.causal_walks_dict = self.causal_walker(adj, Con_Us)
            self.built_cw_adj()
        # 基于RW
        elif args.walk_type == 'RW':
            n_nodes = adj.shape[0]
            self.causal_walks_list, self.causal_walks_dict = RandomWalk(adj, n_nodes, self.walk_length).random_walks()
            self.built_cw_adj()
        # 基于N2VRW
        elif args.walk_type == 'N2VRW':
            n_nodes = adj.shape[0]
            self.causal_walks_list, self.causal_walks_dict = \
                N2V_RandomWalk(adj, n_nodes, self.p, self.q, self.walk_length).node2vec_random_walks()
            self.built_cw_adj()
        else:
            self.causal_walks_list, self.causal_walks_dict = None, None

        self.projections = nn.ModuleList([ResNet(in_dim, h_feats) for in_dim in in_feats])
        if self.is_hetero:
            self.hgts = GNNs.HGT(node_dict, edge_dict, h_feats, h_feats, out_feats, n_layers, n_headers, use_norm=True)
            self.predictor = HeteroDotProductPredictor()
        else:
            self.hgts = HGTConv(h_feats, out_feats, n_headers, 3, 6)
            self.predictor = DotProductPredictor()
        self.init_params()

    def built_cw_adj(self):
        if type(self.causal_walks_list) is list:  # 多重路径的情况
            for i in range(len(self.causal_walks_list)):
                multi_neighbors = self.causal_walks_list[i]
                src = multi_neighbors[0][0]
                for j in range(len(multi_neighbors)):
                    dst = multi_neighbors[j][1:]
                    self.cus_adj[src, dst] = 1
        else:
            for row in range(self.causal_walks_list.shape[0]):
                src = self.causal_walks_list[row][0]
                dsts = self.causal_walks_list[row][1:]
                for dst in dsts:
                    self.cus_adj[src, dst] = 1

    def return_walk_dict(self):
        return self.causal_walks_dict

    def build_custom_graph(self, neighbor_list, is_hetero=True):
        n_nodes = self.adj.shape[0]
        n_lnc, n_mi, n_stress = args.num_lnc, args.num_mi, args.num_stress
        if neighbor_list is None:
            # 不使用游走子图，只使用GNN
            if is_hetero:
                lm_arr = np.nonzero(self.adj[:n_lnc, n_lnc:n_lnc + n_mi])
                ls_arr = np.nonzero(self.adj[:n_lnc, n_lnc + n_mi:])
                ms_arr = np.nonzero(self.adj[n_lnc:n_lnc + n_mi, n_lnc + n_mi:])
                ll_arr = np.nonzero(self.adj[:n_lnc, :n_lnc])
                mm_arr = np.nonzero(self.adj[n_lnc:n_lnc + n_mi, n_lnc:n_lnc + n_mi])
                ss_arr = np.nonzero(self.adj[n_lnc + n_mi:, n_lnc + n_mi:])
                graph = GNN_Hetero_G(ls_arr, ms_arr, lm_arr, ll_arr, mm_arr, ss_arr)
            else:
                src, dst = np.nonzero(self.adj)
                graph = dgl.graph((src, dst), num_nodes=n_nodes)
                graph = dgl.add_self_loop(graph)
        else:
            if is_hetero:
                lm_arr = np.nonzero(self.cus_adj[:n_lnc, n_lnc:n_lnc + n_mi])
                ls_arr = np.nonzero(self.adj[:n_lnc, n_lnc + n_mi:])
                ms_arr = np.nonzero(self.adj[n_lnc:n_lnc + n_mi, n_lnc + n_mi:])
                ll_arr = np.nonzero(self.cus_adj[:n_lnc, :n_lnc])
                mm_arr = np.nonzero(self.cus_adj[n_lnc:n_lnc + n_mi, n_lnc:n_lnc + n_mi])
                ss_arr = np.nonzero(self.cus_adj[n_lnc + n_mi:, n_lnc + n_mi:])
                graph = GNN_Hetero_G(ls_arr, ms_arr, lm_arr, ll_arr, mm_arr, ss_arr)
            else:
                lm_arr = np.nonzero(self.cus_adj[:n_lnc, n_lnc:n_lnc + n_mi])
                ls_arr = np.nonzero(self.adj[:n_lnc, n_lnc + n_mi:])
                ms_arr = np.nonzero(self.adj[n_lnc:n_lnc + n_mi, n_lnc + n_mi:])
                ll_arr = np.nonzero(self.cus_adj[:n_lnc, :n_lnc])
                mm_arr = np.nonzero(self.cus_adj[n_lnc:n_lnc + n_mi, n_lnc:n_lnc + n_mi])
                ss_arr = np.nonzero(self.cus_adj[n_lnc + n_mi:, n_lnc + n_mi:])
                graph = dgl.graph((np.concatenate([lm_arr[0], ls_arr[0], ms_arr[0], ll_arr[0], mm_arr[0], ss_arr[0]]),
                                   np.concatenate([lm_arr[1], ls_arr[1], ms_arr[1], ll_arr[1], mm_arr[1], ss_arr[1]])),
                                  num_nodes=self.adj.shape[0])
                # graph = dgl.add_self_loop(graph)
        return graph

    def a(self):
        self.ntype = torch.cat(
            [torch.zeros(self.n_lnc), torch.ones(self.n_mi), torch.ones(self.n_stress) + 1]).long().to(device)
        num_etypes = [
            int(np.sum(self.cus_adj[:self.n_lnc, self.n_lnc:self.n_lnc + self.n_mi])),
            int(np.sum(self.adj[:self.n_lnc, self.n_lnc + self.n_mi:])),
            int(np.sum(self.adj[self.n_lnc:self.n_lnc + self.n_mi, self.n_lnc + self.n_mi:])),
            int(np.sum(self.cus_adj[:self.n_lnc, :self.n_lnc])),
            int(np.sum(self.cus_adj[self.n_lnc:self.n_lnc + self.n_mi, self.n_lnc:self.n_lnc + self.n_mi])),
            int(np.sum(self.cus_adj[self.n_lnc + self.n_mi:, self.n_lnc + self.n_mi:]))
        ]
        self.etype = torch.cat(
            [torch.zeros(num_etypes[0]),
             torch.zeros(num_etypes[1]) + 1,
             torch.zeros(num_etypes[2]) + 2,
             torch.zeros(num_etypes[3]) + 3,
             torch.zeros(num_etypes[4]) + 4,
             torch.zeros(num_etypes[5]) + 5,
             ]).long().to(device)  # 0, 1, 2 for different edge types

    @staticmethod
    def construct_graph(adj):
        src, dst = np.nonzero(adj)
        graph = dgl.graph((src, dst), num_nodes=adj.shape[0])
        return graph

    def forward(self, adj, neg_adj, X, my_pos_g, my_neg_g, pre_etype):
        projected_emb = {}
        for j, (k, v) in enumerate(X.items()):
            projection = self.projections[j]
            emb = projection(v.float().to(device))
            projected_emb[k] = emb

        if pre_etype:
            causal_custom_g = self.build_custom_graph(self.causal_walks_list, is_hetero=self.is_hetero)
            causal_custom_g = causal_custom_g.to(device)
            node_dict = {}
            edge_dict = {}
            for ntype in my_pos_g.ntypes:
                node_dict[ntype] = len(node_dict)
            for etype in my_pos_g.etypes:
                edge_dict[etype] = len(edge_dict)
                eids = torch.ones(causal_custom_g.num_edges(etype), dtype=torch.long).to(device)
                causal_custom_g.edges[etype].data["id"] = (eids * edge_dict[etype])
            causal_custom_g.ndata['h'] = projected_emb
            if self.is_hetero:
                h, meta_attn_scores = self.hgts(causal_custom_g)
                pos_preds = self.predictor(my_pos_g.to(device), h, pre_etype)
                neg_preds = self.predictor(my_neg_g.to(device), h, pre_etype)
                return pos_preds, neg_preds, self.causal_walks_dict, meta_attn_scores
            else:
                my_pos_g, my_neg_g = self.construct_graph(adj), self.construct_graph(neg_adj)
                my_pos_g = my_pos_g.to(device)
                my_neg_g = my_neg_g.to(device)
                homo_projected_emb = torch.cat([projected_emb['lnc'], projected_emb['mi'], projected_emb['stress']])
                homo_causal_custom_g = dgl.to_homogeneous(causal_custom_g)
                homo_causal_custom_g.ndata['h'] = homo_projected_emb
                h = self.hgts(homo_causal_custom_g, homo_projected_emb.to(device), self.ntype, self.etype)
                pos_preds = self.predictor(my_pos_g, h)
                neg_preds = self.predictor(my_neg_g, h)
                return pos_preds, neg_preds, self.causal_walks_dict, None

    def init_params(self):
        for lin in self.projections:
            try:
                lin.reset_parameters()
            except:
                continue
        try:
            self.gcns.reset_params()
        except:
            pass
        try:
            self.hgts.reset_params()
        except:
            pass
        self.mlp_predictor.reset_params()
