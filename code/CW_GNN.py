# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/19 16:23
@Auth ： He Yu
@File ：CW_GNN.py
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
from dgl.dataloading import negative_sampler

device = torch.device('cuda')
"""
每次epoch都需要重新采样
每次重新采样后，构建新的图，GCN在此图上进行卷积
[[S_lnc_lnc, A_lnc_mi, A_lnc_stress],
 [A_lnc_mi.T, S_mi_mi, A_mi_stress.T],
 [A_lnc_stress.T, A_mi_stress, S_stress_stress]]
"""
args = set_params.get_args()


def temp_Hetero_G(ls, ms, lm, ll, mm, ss):
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


def temp_Hetero_G1(ls, ll, ss):
    hetero_g = dgl.heterograph({
        ('lnc', 'l_s', 'stress'): (
            ls[0], ls[1]
        ),
        ('stress', 's_l', 'lnc'): (
            ls[1], ls[0]
        ),
        ('lnc', 'l_l', 'lnc'): (
            ll
        ),
        ('stress', 's_s', 'stress'): (
            ss
        )
    })
    return hetero_g


def temp_Hetero_abG1(ls):
    hetero_g = dgl.heterograph({
        ('lnc', 'l_s', 'stress'): (
            ls[0], ls[1]
        ),
        ('stress', 's_l', 'lnc'): (
            ls[1], ls[0]
        ),
    })
    num_stress_nodes = hetero_g.num_nodes('stress')
    num_lnc_nodes = hetero_g.num_nodes('lnc')
    if num_stress_nodes < 50:
        # 如果 'stress' 节点数量不足，补充节点
        extra_stress_nodes = 50 - num_stress_nodes
        hetero_g.add_nodes(extra_stress_nodes, ntype='stress')
    if num_lnc_nodes < 175:
        # 如果 'lnc' 节点数量不足，补充节点
        extra_lnc_nodes = 175 - num_lnc_nodes
        hetero_g.add_nodes(extra_lnc_nodes, ntype='lnc')
    return hetero_g


def Hetero_abG1(ls, ms, lm, mm, ss):
    # Hetero_abG1(ls_arr, ms_arr, lm_arr, mm_arr, ss_arr)
    # Hetero_abG1(ls_arr, ms_arr, lm_arr, ll_arr, ss_arr)
    # Hetero_abG1(ls_arr, ms_arr, lm_arr, ll_arr, mm_arr)
    # Hetero_abG1(ls_arr, ms_arr, ll_arr, mm_arr, ss_arr)
    # Hetero_abG1(ls_arr, lm_arr, ll_arr, mm_arr, ss_arr)
    hetero_g = dgl.heterograph({
        ('lnc', 'l_s', 'stress'): (
            ls[0], ls[1]
        ),
        ('stress', 's_l', 'lnc'): (
            ls[1], ls[0]
        ),
        # ('mi', 'm_s', 'stress'): (
        #     ms[0], ms[1]
        # ),
        # ('stress', 's_m', 'mi'): (
        #     ms[1], ms[0]
        # ),
        ('lnc', 'l_m', 'mi'): (
            ms[0], ms[1]
        ),
        ('mi', 'm_l', 'lnc'): (
            ms[1], ms[0]
        ),
        ('lnc', 'l_l', 'lnc'): (
            lm
        ),
        ('mi', 'm_m', 'mi'): (
            mm
        ),
        ('stress', 's_s', 'stress'): (
            ss
        )
    })
    num_stress_nodes = hetero_g.num_nodes('stress')
    num_lnc_nodes = hetero_g.num_nodes('lnc')
    if num_stress_nodes < 50:
        # 如果 'stress' 节点数量不足，补充节点
        extra_stress_nodes = 50 - num_stress_nodes
        hetero_g.add_nodes(extra_stress_nodes, ntype='stress')
    if num_lnc_nodes < 175:
        # 如果 'lnc' 节点数量不足，补充节点
        extra_lnc_nodes = 175 - num_lnc_nodes
        hetero_g.add_nodes(extra_lnc_nodes, ntype='lnc')
    return hetero_g


def temp_Hetero_G2(ms, mm, ss):
    hetero_g = dgl.heterograph({
        ('mi', 'm_s', 'stress'): (
            ms[0], ms[1]
        ),
        ('stress', 's_m', 'mi'): (
            ms[1], ms[0]
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


class Hadmard(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            message = fn.u_mul_v('h', 'h', 'z')
            graph.apply_edges(message, etype=etype)
            return graph.edges[etype].data['z']


class Concatenate(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            # Assign node features to graph
            graph.ndata['h'] = h

            # Define the concatenation operation as the message function
            def concat_message(edges):
                # Concatenate the source and destination node features
                return {'z': torch.cat([edges.src['h'], edges.dst['h']], dim=-1)}

            # Apply the message function to the edges
            graph.apply_edges(concat_message, etype=etype)

            # Return the concatenated edge features
            return graph.edges[etype].data['z']


class MLP_Predictor(nn.Module):
    def __init__(self, dim_z):
        super(MLP_Predictor, self).__init__()
        dim_in = dim_z
        self.mlp_out = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.BatchNorm1d(dim_in // 2),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(dim_in // 2, 1, bias=False),
        )

    def forward(self, z):
        h = self.mlp_out(z).squeeze()
        return h

    def reset_params(self):
        for lin in self.mlp_out:
            try:
                lin.reset_parameters()
            except:
                continue


class CW_GNN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, adj, Con_Us, p, q, walk_length, window_size,
                 node_dict, edge_dict, n_layers, n_headers, is_hetero, multi_n=1):
        super(CW_GNN, self).__init__()
        self.adj = adj
        self.Con_Us = Con_Us
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.n_layers = n_layers
        self.n_headers = n_headers
        self.is_hetero = is_hetero
        self.multi_n = multi_n
        # self.causal_walker = MultiCausalWalk(p, q, walk_length, window_size, multi_n)
        # 基于CW
        if args.walk_type == 'CW':
            self.causal_walker = MultiCausalWalk(p, q, walk_length, window_size, multi_n)
            self.causal_walks_list, self.causal_walks_dict = self.causal_walker(adj, Con_Us)
        # 基于RW
        elif args.walk_type == 'RW':
            n_nodes = adj.shape[0]
            self.causal_walks_list, self.causal_walks_dict = RandomWalk(adj, n_nodes, self.walk_length).random_walks()
        # 基于N2VRW
        elif args.walk_type == 'N2VRW':
            n_nodes = adj.shape[0]
            self.causal_walks_list, self.causal_walks_dict = \
                N2V_RandomWalk(adj, n_nodes, self.p, self.q, self.walk_length).node2vec_random_walks()
        else:
            self.causal_walks_list, self.causal_walks_dict = None, None

        self.projections = nn.ModuleList([ResNet(in_dim, h_feats) for in_dim in in_feats])

        if is_hetero:
            self.hgts = GNNs.HGT(node_dict, edge_dict, h_feats, h_feats, out_feats, n_layers, n_headers, use_norm=True)
            self.predictor = HeteroDotProductPredictor()
        else:
            # self.gcns = GNNs.dglHGT(h_feats, out_feats, n_headers, 3, 6, 3)
            self.gcns = GNNs.GCN(h_feats, h_feats, out_feats, 0.2)
            # self.gcns = GNNs.GIN(h_feats, h_feats, out_feats, 0.2)
            self.predictor = DotProductPredictor()
        self.hadmard = Hadmard()
        self.concatenate = Concatenate()
        self.mlp_predictor = MLP_Predictor(out_feats)
        self.init_params()

    def return_walk_dict(self):
        return self.causal_walks_dict

    def build_custom_graph(self, neighbor_list, is_hetero=True):
        n_nodes = self.adj.shape[0]
        self.cus_adj = np.zeros_like(self.adj)
        n_lnc, n_mi, n_stress = args.num_lnc, args.num_mi, args.num_stress
        if neighbor_list is None:
            # 不使用游走子图，只使用GNN
            if is_hetero:
                if args.DataClass == 'lnc_mi_stress_graphs':
                    lm_arr = np.nonzero(self.adj[:n_lnc, n_lnc:n_lnc + n_mi])
                    ls_arr = np.nonzero(self.adj[:n_lnc, n_lnc + n_mi:])
                    ms_arr = np.nonzero(self.adj[n_lnc:n_lnc + n_mi, n_lnc + n_mi:])
                    ll_arr = np.nonzero(self.adj[:n_lnc, :n_lnc])
                    mm_arr = np.nonzero(self.adj[n_lnc:n_lnc + n_mi, n_lnc:n_lnc + n_mi])
                    ss_arr = np.nonzero(self.adj[n_lnc + n_mi:, n_lnc + n_mi:])
                    graph = temp_Hetero_G(ls_arr, ms_arr, lm_arr, ll_arr, mm_arr, ss_arr)
                elif args.DataClass == 'lnc_stress_graphs':
                    ls_arr = np.nonzero(self.adj[:n_lnc, n_lnc:])
                    ll_arr = np.nonzero(self.adj[:n_lnc, :n_lnc])
                    ss_arr = np.nonzero(self.adj[n_lnc:, n_lnc:])
                    graph = temp_Hetero_G1(ls_arr, ll_arr, ss_arr)
                else:
                    ms_arr = np.nonzero(self.adj[:n_mi, n_mi:])
                    mm_arr = np.nonzero(self.adj[:n_mi, :n_mi])
                    ss_arr = np.nonzero(self.adj[n_mi:, n_mi:])
                    graph = temp_Hetero_G2(ms_arr, mm_arr, ss_arr)
            else:
                src, dst = np.nonzero(self.adj)
                graph = dgl.graph((src, dst), num_nodes=n_nodes)
                graph = dgl.add_self_loop(graph)
        else:
            if type(neighbor_list) is list:  # 多重路径的情况
                for i in range(len(neighbor_list)):
                    multi_neighbors = neighbor_list[i]
                    src = multi_neighbors[0][0]
                    for j in range(len(multi_neighbors)):
                        dst = multi_neighbors[j][1:]
                        self.cus_adj[src, dst] = 1
            else:
                for row in range(neighbor_list.shape[0]):
                    src = neighbor_list[row][0]
                    dsts = neighbor_list[row][1:]
                    for dst in dsts:
                        self.cus_adj[src, dst] = 1
            if is_hetero:
                if args.DataClass == 'lnc_mi_stress_graphs':
                    lm_arr = np.nonzero(self.cus_adj[:n_lnc, n_lnc:n_lnc + n_mi])
                    ls_arr = np.nonzero(self.adj[:n_lnc, n_lnc + n_mi:])
                    ms_arr = np.nonzero(self.adj[n_lnc:n_lnc + n_mi, n_lnc + n_mi:])
                    ll_arr = np.nonzero(self.cus_adj[:n_lnc, :n_lnc])
                    mm_arr = np.nonzero(self.cus_adj[n_lnc:n_lnc + n_mi, n_lnc:n_lnc + n_mi])
                    ss_arr = np.nonzero(self.cus_adj[n_lnc + n_mi:, n_lnc + n_mi:])
                    graph = temp_Hetero_G(ls_arr, ms_arr, lm_arr, ll_arr, mm_arr, ss_arr)
                    # graph = Hetero_abG1(ls_arr, ms_arr, lm_arr, mm_arr, ss_arr)
                    # graph = Hetero_abG1(ls_arr, ms_arr, lm_arr, ll_arr, ss_arr)
                    # graph = Hetero_abG1(ls_arr, ms_arr, lm_arr, ll_arr, mm_arr)
                    # graph = Hetero_abG1(ls_arr, ms_arr, ll_arr, mm_arr, ss_arr)
                    # graph = Hetero_abG1(ls_arr, lm_arr, ll_arr, mm_arr, ss_arr)
                    # graph = Hetero_abG1(ls_arr, lm_arr, ll_arr, mm_arr, ss_arr)
                # # 仅使用lncRNA-stress数据
                elif args.DataClass == 'lnc_stress_graphs':
                    ls_arr = np.nonzero(self.adj[:n_lnc, n_lnc:])
                    ll_arr = np.nonzero(self.cus_adj[:n_lnc, :n_lnc])
                    ss_arr = np.nonzero(self.cus_adj[n_lnc:, n_lnc:])
                    graph = temp_Hetero_G1(ls_arr, ll_arr, ss_arr)
                    # graph = temp_Hetero_abG1(ls_arr)
                # 仅使用miRNA-stress数据
                else:
                    ms_arr = np.nonzero(self.adj[:n_mi, n_mi:])
                    mm_arr = np.nonzero(self.cus_adj[:n_mi, :n_mi])
                    ss_arr = np.nonzero(self.cus_adj[n_mi:, n_mi:])
                    graph = temp_Hetero_G2(ms_arr, mm_arr, ss_arr)
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

    @staticmethod
    def construct_graph(adj):
        src, dst = np.nonzero(adj)
        graph = dgl.graph((src, dst), num_nodes=adj.shape[0])
        return graph

    def forward(self, adj, neg_adj, X, my_pos_g, my_neg_g, pre_etype):
        # 传入的图为因果采样图，以及节点特征
        # 在此处传入原始的图，并在此进行采样，构建因果采样图
        # causal_walk_embs的形状为nxh，n表示节点个数，h表示hidden dim
        # 基于CW
        # if args.walk_type == 'CW':
        #     self.causal_walks_list, self.causal_walks_dict = self.causal_walker(adj, self.Con_Us)
        # # 基于RW
        # elif args.walk_type == 'RW':
        #     n_nodes = adj.shape[0]
        #     self.causal_walks_list, self.causal_walks_dict = RandomWalk(adj, n_nodes, self.walk_length).random_walks()
        # # 基于N2VRW
        # elif args.walk_type == 'N2VRW':
        #     n_nodes = adj.shape[0]
        #     self.causal_walks_list, self.causal_walks_dict = \
        #         N2V_RandomWalk(adj, n_nodes, self.p, self.q, self.walk_length).node2vec_random_walks()
        # # 4. 不使用随机游走，仅使用图神经网络
        # else:
        #     self.causal_walks_list, self.causal_walks_dict = None, None
        # 构建因果采样图
        n_lnc, n_mi, n_stress = args.num_lnc, args.num_mi, args.num_stress
        if pre_etype and self.is_hetero:
            causal_custom_g = self.build_custom_graph(self.causal_walks_list, is_hetero=self.is_hetero)
            causal_custom_g = causal_custom_g.to(device)
            # node_dict = {}
            # edge_dict = {}
            # for ntype in my_pos_g.ntypes:
            #     node_dict[ntype] = len(node_dict)
            # for etype in my_pos_g.etypes:
            #     edge_dict[etype] = len(edge_dict)
            #     eids = torch.ones(causal_custom_g.num_edges(etype), dtype=torch.long).to(device)
            #     causal_custom_g.edges[etype].data["id"] = (eids * edge_dict[etype])
        else:
            causal_custom_g = self.build_custom_graph(self.causal_walks_list, is_hetero=self.is_hetero)
            causal_custom_g = causal_custom_g.to(device)

        projected_emb = {}
        for j, (k, v) in enumerate(X.items()):
            projection = self.projections[j]
            emb = projection(v.float().to(device))
            projected_emb[k] = emb

        if self.is_hetero:
            causal_custom_g.ndata['h'] = projected_emb
            h = self.hgts(causal_custom_g)
            pos_preds = self.predictor(my_pos_g.to(device), h, pre_etype)
            neg_preds = self.predictor(my_neg_g.to(device), h, pre_etype)
            return pos_preds, neg_preds, self.causal_walks_dict
        else:
            pos_g, neg_g = self.construct_graph(adj), self.construct_graph(neg_adj)
            pos_g = pos_g.to(device)
            neg_g = neg_g.to(device)
            if args.DataClass == 'lnc_mi_stress_graphs':
                homo_projected_emb = torch.cat([projected_emb['lnc'], projected_emb['mi'], projected_emb['stress']])
            elif args.DataClass == 'lnc_stress_graphs':
                homo_projected_emb = torch.cat([projected_emb['lnc'], projected_emb['stress']])
            else:
                homo_projected_emb = torch.cat([projected_emb['mi'], projected_emb['stress']])
            #
            # self.ntype = torch.cat(
            #     [torch.zeros(175), torch.ones(500), torch.ones(50) + 1]).long().to(device)  # 0 for user, 1 for item
            # num_etypes = [
            #     int(np.sum(self.cus_adj[:n_lnc, n_lnc:n_lnc + n_mi])),
            #     # int(np.sum(self.cus_adj[:n_lnc, n_lnc:n_lnc + n_mi])),
            #     int(np.sum(self.adj[:n_lnc, n_lnc + n_mi:])),
            #     # int(np.sum(self.adj[:n_lnc, n_lnc + n_mi:])),
            #     int(np.sum(self.adj[n_lnc:n_lnc + n_mi, n_lnc + n_mi:])),
            #     # int(np.sum(self.adj[n_lnc:n_lnc + n_mi, n_lnc + n_mi:])),
            #     int(np.sum(self.cus_adj[:n_lnc, :n_lnc])),
            #     int(np.sum(self.cus_adj[n_lnc:n_lnc + n_mi, n_lnc:n_lnc + n_mi])),
            #     int(np.sum(self.cus_adj[n_lnc + n_mi:, n_lnc + n_mi:]))
            # ]
            #
            # self.etype = torch.cat(
            #     [torch.zeros(num_etypes[0]),
            #      torch.zeros(num_etypes[1]) + 1,
            #      torch.zeros(num_etypes[2]) + 2,
            #      torch.zeros(num_etypes[3]) + 3,
            #      torch.zeros(num_etypes[4]) + 4,
            #      torch.zeros(num_etypes[5]) + 5,
            #      # torch.zeros(num_etypes[6]) + 6,
            #      # torch.zeros(num_etypes[7]) + 7,
            #      # torch.zeros(num_etypes[8]) + 8,
            #      ]).long().to(device)  # 0, 1, 2 for different edge types
            # h = self.gcns(homo_causal_custom_g, homo_projected_emb.to(device), self.ntype, self.etype)
            h = self.gcns(causal_custom_g, homo_projected_emb)
            pos_preds = self.predictor(pos_g, h)
            neg_preds = self.predictor(neg_g, h)
            return pos_preds, neg_preds, self.causal_walks_dict, None
        # h = torch.cat([projected_emb['lnc'], projected_emb['mi'], projected_emb['stress']])
        # h = self.gcns(causal_custom_g, h)

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
