# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/5 15:19
@Auth ： He Yu
@File ：test_CausalWalk.py
@IDE ：PyCharm
@Function ：Function of the script

目标：为每个类型的节点进行因果游走得到游走序列
基本步骤：
得到整个异构网络的邻接矩阵
在整个异构网络上进行游走
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""
import c_utils
import utils
import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from multiprocessing import Pool
from sknetwork.clustering import Louvain
from bulit_graphs import lnc_mi_stress_graphs
import json
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from dgl.sampling import node2vec_random_walk, random_walk


class RandomWalk(object):
    def __init__(self, adj, n_nodes, walk_length):
        self.adj = adj
        self.n_nodes = n_nodes
        self.nodes = [i for i in range(n_nodes)]
        self.walk_length = walk_length
        src, dst = np.nonzero(adj)
        self.g = dgl.graph((src, dst), num_nodes=self.n_nodes)

    def random_walks(self):
        walks = random_walk(self.g, self.nodes, length=self.walk_length)
        walks = walks[0].numpy()
        walks = np.concatenate(
            [np.array(self.nodes).reshape(-1, 1), walks], axis=1
        )
        walks_dict = {i: walk for i, walk in enumerate(walks)}
        return walks, walks_dict


class N2V_RandomWalk(object):
    def __init__(self, adj, n_nodes, p, q, walk_length):
        self.adj = adj
        self.n_nodes = n_nodes
        self.nodes = [i for i in range(n_nodes)]
        self.p = p
        self.q = q
        self.walk_length = walk_length
        src, dst = np.nonzero(adj)
        self.g = dgl.graph((src, dst), num_nodes=self.n_nodes)

    def node2vec_random_walks(self):
        walks = node2vec_random_walk(self.g, self.nodes, self.p, self.q, self.walk_length)
        walks = walks.numpy()
        walks = np.concatenate(
            [np.array(self.nodes).reshape(-1, 1), walks], axis=1
        )
        walks_dict = {i: walk for i, walk in enumerate(walks)}
        return walks, walks_dict


class CausalWalk(object):
    def __init__(self, net, con, p=1, q=1, walk_length=100):
        self._net = net
        self._walk_length = walk_length
        self.p = p
        self.q = q
        self.con = con
        self.neighbor_table = {}
        self.num_node = self._net.shape[0]
        self.Y = self.get_community_labels(self._net)
        self.max_Y = np.max(self.Y)
        for i in range(len(self._net)):
            neighbors = []
            for j in range(len(self._net[i])):
                if self._net[i][j] != 0:
                    neighbors.append(j)
            self.neighbor_table[i] = neighbors
        # print(self.neighbor_table)

    @staticmethod
    def get_community_labels(adj):
        louvain = Louvain()
        labels = louvain.fit_transform(adj)
        return labels.astype(np.int32)

    def causal_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self._walk_length:
            current_node = walk[-1]
            neighbors = self.neighbor_table[current_node]
            if len(neighbors) == 0:
                # Remove current_node and reselect start_node if there are no neighbors
                walk.pop()
                if len(walk) == 0:  # If the walk is empty after pop, select a new start node
                    start_node = np.random.choice(list(self.neighbor_table.keys()))
                    walk.append(start_node)
                continue
            label_Y = self.Y[neighbors]
            # label_Y = list(label_Y)
            max_Y = self.max_Y
            label_yr = np.unique(label_Y).astype(np.int32)
            con_us = []
            for i in neighbors:
                con_us.append(self.con[i])
            con_ur = np.unique(con_us).astype(np.int32)
            neighbors = list(neighbors)
            con_us = list(con_us)
            causal_sample_dict = c_utils.cb_backdoor(neighbors, label_Y, con_us, label_yr, con_ur, len(neighbors),
                                                     max_Y)
            # print(causal_sample_dict)
            if len(walk) == 1:
                probs = c_utils.compute_transition_probs(self.neighbor_table, self.p, self.q, causal_sample_dict)
            else:
                prev_node = walk[-2]
                probs = c_utils.compute_transition_probs(self.neighbor_table, self.p, self.q, causal_sample_dict,
                                                         prev_node)
            probs /= np.sum(probs)
            max_probs = np.max(probs)
            max_probs_idx = np.argmax(probs)
            # print(neighbors)
            # next_node = np.random.choice(neighbors, p=probs.reshape(-1, ))
            next_node = neighbors[max_probs_idx]
            walk.append(next_node)
            # print(walk)
        return walk

    # def causal_path(self):
    #     walks = {}
    #     for i in range(self.num_node):
    #         walk = self.causal_walk(i)
    #         walks[i] = walk
    #     return walks

    def causal_path(self):
        with Pool(processes=8) as pool:
            walks = pool.map(self.causal_walk, range(self.num_node))
        # 将列表转换为字典
        walks_dict = {i: walk for i, walk in enumerate(walks)}
        return walks_dict

    def single_causal_path(self, node_index, rewalk_num):
        tmp_walks = []
        for k in range(rewalk_num):
            walk = self.causal_walk(node_index)
            tmp_walks.append(walk)
        return node_index, tmp_walks

    def multi_causal_path(self, rewalk_num):
        with multiprocessing.Pool(processes=2) as pool:
            results = pool.starmap(self.single_causal_path, [(i, rewalk_num) for i in range(self.num_node)])
        multi_walks = {node_index: walks for node_index, walks in results}
        return multi_walks


class MultiCausalWalk(nn.Module):
    def __init__(self, p, q, walk_length, window_size, multi_n=1):
        super(MultiCausalWalk, self).__init__()
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.window_size = window_size
        self.multi_n = multi_n

    def forward(self, adj, Con_Us):
        c_walker = CausalWalk(adj, Con_Us, self.p, self.q, walk_length=self.walk_length)
        multi_paths = c_walker.multi_causal_path(self.multi_n)
        multi_walks = list(multi_paths.values())
        return multi_walks, multi_paths


if __name__ == '__main__':
    data = lnc_mi_stress_graphs.MYdataset()
    hg = data[0]
    node_feats = hg.ndata['h']
    lnc_feats = node_feats['lnc']
    mi_feats = node_feats['mi']
    stress_feats = node_feats['stress']
