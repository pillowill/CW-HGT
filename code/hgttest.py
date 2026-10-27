# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/15 20:30
@Auth ： He Yu
@File ：hgttest.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import torch
import dgl
from dgl.nn.pytorch import HGTConv

# Sample heterogeneous graph
data_dict = {
    ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('user', 'clicks', 'item'): (torch.tensor([0, 2]), torch.tensor([2, 3])),
    ('item', 'bought_by', 'user'): (torch.tensor([3, 1]), torch.tensor([1, 0])),
}
g = dgl.heterograph(data_dict)

# Node features (for simplicity, we'll use random features)
# Let's assume all node types have a feature dimension of 10
user_features = torch.randn(3, 10)  # 3 users
item_features = torch.randn(4, 10)  # 4 items

# Assigning node features for each type
features = torch.cat([user_features, item_features], dim=0)

# Node and edge types (mapping to integers for simplicity)
ntype = torch.cat([torch.zeros(3), torch.ones(4)]).long()  # 0 for user, 1 for item
etype = torch.cat([torch.zeros(2), torch.ones(2), torch.full((2,), 2)]).long()  # 0, 1, 2 for different edge types

# Convert the heterogeneous graph to homogeneous form
hg = dgl.to_homogeneous(g)
hg.ndata['h'] = features

# Define the HGTConv layer
in_size = 10  # input feature size
head_size = 8  # size of each attention head
num_heads = 4  # number of attention heads
num_ntypes = 2  # number of node types ('user', 'item')
num_etypes = 3  # number of edge types ('follows', 'clicks', 'bought_by')

hgt_layer = HGTConv(in_size, head_size, num_heads, num_ntypes, num_etypes)

# Forward pass through the HGT layer
node_emb = hgt_layer(hg, hg.ndata['h'], ntype, etype)

# Output node embeddings
print(node_emb)
