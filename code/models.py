# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/28 15:45
@Auth ： He Yu
@File ：models.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import dgl
import dgl.nn.pytorch as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm
import dgl.function as fn
from dgl.nn.pytorch import GATConv


class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(in_dim[i], out_dim, bias=False) for i in range(len(in_dim))])
        # 使用多层网络结构
        self.bn = nn.BatchNorm1d(out_dim)
        self.ac = nn.LeakyReLU()
        # 使用残差连接
        self.shortcuts = nn.ModuleList([nn.Linear(in_dim[i], out_dim) for i in range(len(in_dim))])
        self.reset_params()

    def forward(self, in_feats):
        outs = []
        for i in range(len(in_feats)):
            x = self.linears[i](in_feats[i])
            x = self.bn(x)
            x = self.ac(x)
            res = self.shortcuts[i](in_feats[i])
            out = torch.add(x, res)
            outs.append(out)
        return outs

    def reset_params(self):
        for linear, shortcut in zip(self.linears, self.shortcuts):
            linear.reset_parameters()
            shortcut.reset_parameters()
        self.bn.reset_parameters()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self):


        """
        The main framework of model is divided into 2 parts:
        1. the gnns
        2. the casual walk
        :return:
        """

        # 创建一个包含 4 个节点和 5 条边的示例图
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        g = dgl.graph(edges)

        # 定义边的概率特征
        probabilities = torch.tensor([0.2, 0.1, 0.4, 0.2, 0.1], dtype=torch.float32)
        g.edata['prob'] = probabilities

        # 参数定义
        p = 1.0  # 返回参数
        q = 1.0  # 出发参数
        walk_length = 5  # 随机游走长度

        # 选择起始节点
        start_nodes = torch.tensor([0, 1, 2, 3])

        # 执行 Node2Vec 随机游走，使用 'prob' 作为概率特征
        walks = dgl.sampling.node2vec_random_walk(g, start_nodes, p, q, walk_length, prob='prob')

        print(walks)

        pass
