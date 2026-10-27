# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/12 21:21
@Auth ： He Yu
@File ：MetaRelationAttentionAnalysis.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import numpy as np
import torch
import copy
import set_params
from utils import *
from CW_GNN import CW_GNN
from bulit_graphs import lnc_mi_stress_graphs
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib import gridspec


def draw_learned_meta_path(list1, list2):
    plt.rc('font', family='Times New Roman')
    # 创建画布并设置为 3 行 5 列
    x_labels = ['a', 'b', 'c', 'd', 'e']
    learned_meta_path = [
        'lncRNA-lncRNA-stress',
        'stress-lncRNA, miRNA-stress',
        'stress-lncRNA-stress',
        'stress-lncRNA-stress',
        'stress-lncRNA-stress',
    ]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()  # 拉平方便索引

    # 设置 color bar 显示格式
    def format_colorbar(value, tick_number):
        return f"{value:.2f}"  # 保留两位小数

    for i in range(5):  # 遍历5个索引
        # 取出每个列表中对应索引的数组
        array1 = list1[i]
        array2 = list2[i]

        # 绘制热力图
        heatmap1 = sns.heatmap(array1, ax=axes[2 * i], cbar=True, annot=False, cmap='Blues',
                               xticklabels=["lncRNA", "miRNA", "stress"],
                               yticklabels=["lncRNA", "miRNA", "stress"],
                               cbar_kws={'shrink': 0.9})
        axes[2 * i].set_title(f"({x_labels[i]}) Meta-path learned from five layers of CW-HGT: {learned_meta_path[i]}",
                              fontsize=20, y=-0.25, x=1.35, ha='center')
        axes[2 * i].set_aspect('equal', adjustable='box')  # 设置热力图为正方形
        colorbar1 = heatmap1.collections[0].colorbar
        colorbar1.formatter = FuncFormatter(format_colorbar)  # 设置 color bar 显示两位小数
        colorbar1.update_ticks()
        colorbar1.ax.tick_params(labelsize=18)  # 增大 color bar 刻度字体大小
        axes[2 * i].tick_params(axis='both', which='major', labelsize=18)
        heatmap2 = sns.heatmap(array2, ax=axes[2 * i + 1], cbar=True, annot=False, cmap='Blues',
                               xticklabels=["lncRNA", "miRNA", "stress"],
                               yticklabels=["lncRNA", "miRNA", "stress"],
                               cbar_kws={'shrink': 0.9})
        axes[2 * i + 1].set_aspect('equal', adjustable='box')
        # 设置 color bar 格式
        colorbar2 = heatmap2.collections[0].colorbar
        colorbar2.formatter = FuncFormatter(format_colorbar)  # 设置 color bar 显示两位小数
        colorbar2.update_ticks()
        colorbar2.ax.tick_params(labelsize=18)  # 增大 color bar 刻度字体大小
        axes[2 * i + 1].tick_params(axis='both', which='major', labelsize=18)

    for i in range(10, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig('./定量分析/元路径学习图.png', dpi=600)
    plt.show()


args = set_params.get_args()
num_lnc, num_mi, num_stress = args.num_lnc, args.num_mi, args.num_stress
DataClass = args.DataClass

data = lnc_mi_stress_graphs.MYdataset()
hg = data[0]
node_feats = hg.ndata['h']
lnc_feats = node_feats['lnc']
mi_feats = node_feats['mi']
stress_feats = node_feats['stress']
in_dims = [lnc_feats.shape[1], mi_feats.shape[1], stress_feats.shape[1]]

node_dict = {}
edge_dict = {}
for ntype in hg.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in hg.etypes:
    edge_dict[etype] = len(edge_dict)

k_fold = args.kFold
pre_edge_type = args.pre_edge_type
pos_tr_gs, pos_te_gs = data.split_train_test(hg, k_fold, pre_edge_type)
neg_tr_gs, neg_te_gs = data.negative_sample(hg, k=1, kFold=k_fold, pre_edge=pre_edge_type)

h_dim = args.HidDim
out_dim = args.OutDim
p = args.p
q = args.q
num_epochs = args.epochs
lr = args.lr
KNN_K = args.kNN
weight_decay = args.weight_decay

params_ave_test_metrics_dict = {}
params_std_test_metrics_dict = {}

walk_length_list = 5
# walk_length_list = [25]
HGT_layers_list = 2
# HGT_layers_list = [2]
headers_list = 4
# headers_list = [4]

tr_hg = pos_tr_gs[0]
tr_neg_hg = neg_tr_gs[0]
adj = data.get_adj(tr_hg)
neg_adj = data.get_adj(tr_neg_hg)
num_nodes = adj.shape[0]
Con_Us = cal_conUs(num_nodes, seed=3047 - 0)
if args.DataClass == 'lnc_mi_stress_graphs':
    Con_Us_dict = {
        'lnc': Con_Us[:num_lnc], 'mi': Con_Us[num_lnc:num_lnc + num_mi], 'stress': Con_Us[num_lnc + num_mi:]}
elif args.DataClass == 'lnc_stress_graphs':
    Con_Us_dict = {'lnc': Con_Us[:num_lnc], 'stress': Con_Us[num_lnc:]}
else:
    Con_Us_dict = {'mi': Con_Us[:num_mi], 'stress': Con_Us[num_mi:]}
# 构建扰动后的X，并返回到hg中
Xs = hg.ndata['h']
perturbed_Xs = {k: torch.from_numpy(get_perturbation(Con_Us_dict[k], v)) for k, v in Xs.items()}
perturbed_hg = copy.deepcopy(tr_hg)
perturbed_hg.ndata['h'] = perturbed_Xs
# 模型定义参数
# in_feats, h_feats, out_feats,
# adj, Con_Us, p, q, walk_length, window_size
if __name__ == '__main__':
    # device = torch.device('cuda')
    # walk_length, window_size, n_layers, n_headers = 5, 5 * 0.4, 5, 2
    # # model = CW_GNN(in_dims, h_dim, out_dim, adj, Con_Us, p, q, walk_length, window_size,
    # #                node_dict, edge_dict, n_layers=n_layers, n_headers=n_headers,
    # #                multi_n=args.NumMultiWalk, is_hetero=args.isHetero)
    # cv_atts = []
    # cv_msgs = []
    # for i in range(10):
    #     model = CW_GNN(in_dims, h_dim, out_dim, adj, Con_Us, p, q, walk_length, window_size,
    #                    node_dict, edge_dict, n_layers=n_layers, n_headers=n_headers,
    #                    multi_n=args.NumMultiWalk, is_hetero=args.isHetero)
    #     model.load_state_dict(torch.load(f'./定量分析/15_128_128_5_5_2_HGT_{i}.pth'))
    #     model.eval()
    #     model = model.to(device)
    #     pos_score, neg_score, causal_walks_dict = model(adj, neg_adj, perturbed_Xs, tr_hg,
    #                                                     tr_neg_hg, ('lnc', 'l_s', 'stress'))
    #
    #     cv_atts.append(model.hgts.n_layers_att)
    #     cv_msgs.append(model.hgts.n_layers_msg)
    # # 10-cv
    # # 5层HGT
    # n_meta_attns = []
    # n_meta_msgs = []
    # for j in range(5):
    #     meta_attns = np.zeros(shape=(3, 3))
    #     meta_msgs = np.zeros(shape=(3, 3))
    #     for i in range(10):
    #         att_layer = cv_atts[i][j]
    #         msg_layer = cv_msgs[i][j]
    #         n_meta = 9
    #         meta_rel = ['l_l', 'l_m', 'l_s', 'm_l', 'm_m', 'm_s', 's_l', 's_m', 's_s']
    #         meta_attn = np.zeros(shape=(3, 3))
    #         meta_msg = np.zeros(shape=(3, 3))
    #         for k in range(n_meta):
    #             tmp1 = att_layer[k].detach().cpu().numpy()
    #             tmp2 = msg_layer[meta_rel[k]].detach().cpu().numpy()
    #             j_layer_k_meta = np.mean(np.mean(att_layer[k].detach().cpu().numpy(), axis=1))
    #             j_layer_k_msg = np.mean(np.mean(msg_layer[meta_rel[k]].detach().cpu().numpy()))
    #             meta_attn[k // 3, k % 3] = j_layer_k_meta
    #             meta_msg[k // 3, k % 3] = j_layer_k_msg
    #         meta_attns += meta_attn
    #         meta_msgs += meta_msg
    #     meta_attns /= 10
    #     meta_msgs /= 10
    #     n_meta_attns.append(meta_attns)
    #     n_meta_msgs.append(meta_msgs)
    n_meta_attns = np.load('./定量分析/n_meta_attns.npy')
    n_meta_msgs = np.load('./定量分析/n_meta_msgs.npy')
    np.save('./定量分析/n_meta_attns.npy', n_meta_attns)
    np.save('./定量分析/n_meta_msgs.npy', n_meta_msgs)
    draw_learned_meta_path(n_meta_attns, n_meta_msgs)


