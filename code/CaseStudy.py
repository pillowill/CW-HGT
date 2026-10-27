# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/11 14:29
@Auth ： He Yu
@File ：CaseStudy.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import numpy as np
import pandas as pd
import torch
import copy
import set_params
from utils import *
from ranger import Ranger
from CW_GNN import CW_GNN
from bulit_graphs import lnc_mi_stress_graphs
import dgl
from new_main import *


def get_case_graph():
    k_of_knn = args.kNN
    knn_lnc_g = dgl.knn_graph(lnc_feats, k=k_of_knn, dist='cosine')
    ll_src, ll_dst = knn_lnc_g.edges()

    knn_mi_g = dgl.knn_graph(mi_feats, k=k_of_knn, dist='cosine')
    mm_src, mm_dst = knn_mi_g.edges()

    knn_stress_g = dgl.knn_graph(stress_feats, k=k_of_knn, dist='cosine')
    ss_src, ss_dst = knn_stress_g.edges()

    l_s_row, l_s_col = np.nonzero(case_adj[:num_lnc, num_lnc + num_mi:])
    m_s_row, m_s_col = np.nonzero(case_adj[num_lnc:num_lnc + num_mi, num_lnc + num_mi:])
    l_m_row, l_m_col = np.nonzero(case_adj[:num_lnc, num_lnc:num_lnc + num_mi])

    case_hetero_g = dgl.heterograph({
        ('lnc', 'l_s', 'stress'): (
            torch.tensor(l_s_row),
            torch.tensor(l_s_col)
        ),
        ('stress', 's_l', 'lnc'): (
            torch.tensor(l_s_col),
            torch.tensor(l_s_row)
        ),
        ('mi', 'm_s', 'stress'): (
            torch.tensor(m_s_row),
            torch.tensor(m_s_col)
        ),
        ('stress', 's_m', 'mi'): (
            torch.tensor(m_s_col),
            torch.tensor(m_s_row)
        ),
        ('lnc', 'l_m', 'mi'): (
            torch.tensor(l_m_row),
            torch.tensor(l_m_col)
        ),
        ('mi', 'm_l', 'lnc'): (
            torch.tensor(l_m_col),
            torch.tensor(l_m_row)
        ),
        ('lnc', 'l_l', 'lnc'): (
            ll_src, ll_dst
        ),
        ('mi', 'm_m', 'mi'): (
            mm_src, mm_dst
        ),
        ('stress', 's_s', 'stress'): (
            ss_src, ss_dst
        )
    })
    case_hetero_g.ndata['h'] = node_feat_dict
    return case_hetero_g


def id_mapping():
    # 读取 lncid.csv、stressid.csv 和结果文件
    lnc_id = pd.read_csv("./dataset/lnc_ids.csv", header=0, names=["lncRNA_name", "lncRNA_id"])
    stress_id = pd.read_csv("./dataset/stress_ids.csv", header=0, names=["stress_name", "stress_id"])
    result = pd.read_csv("./案例分析/case_final_results1.csv", header=None, names=["lncRNA_id", "stress_id", "score"])

    # 将 lncRNA_id 和 stress_id 分别映射到对应名称
    result = result.merge(lnc_id, on="lncRNA_id", how="left")  # 映射 lncRNA 名称
    result = result.merge(stress_id, on="stress_id", how="left")  # 映射 stress 名称

    # 调整列的顺序
    result = result[["lncRNA_name", "stress_name", "score"]]

    # 保存映射后的结果
    result.to_csv("./案例分析/mapped_result1.csv", index=False)
    print("映射完成，结果已保存到 mapped_result.csv")


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

all_adj = data.get_adj(hg)
neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(1, exclude_self_loops=True)
eid_dict = {etype: hg.edges(etype=etype, form='eid') for etype in
            ['l_s']}
neg_eid_sample_dict = neg_sampler(hg, eid_dict)
neg_eid_sample_dict[('stress', 's_l', 'lnc')] = (neg_eid_sample_dict[('lnc', 'l_s', 'stress')][1],
                                                 neg_eid_sample_dict[('lnc', 'l_s', 'stress')][0])
neg_eid_sample_dict[('lnc', 'l_l', 'lnc')] = hg.edge_type_subgraph(['l_l']).find_edges(
    hg.edges(etype='l_l', form='eid'))
neg_eid_sample_dict['mi', 'm_m', 'mi'] = hg.edge_type_subgraph(['m_m']).find_edges(
    hg.edges(etype='m_m', form='eid'))
neg_eid_sample_dict['stress', 's_s', 'stress'] = hg.edge_type_subgraph(['s_s']).find_edges(
    hg.edges(etype='s_s', form='eid'))
neg_eid_sample_dict['lnc', 'l_m', 'mi'] = hg.edge_type_subgraph(['l_m']).find_edges(
    hg.edges(etype='l_m', form='eid'))
neg_eid_sample_dict['mi', 'm_l', 'lnc'] = hg.edge_type_subgraph(['m_l']).find_edges(
    hg.edges(etype='m_l', form='eid'))
neg_eid_sample_dict['mi', 'm_s', 'stress'] = hg.edge_type_subgraph(['m_s']).find_edges(
    hg.edges(etype='m_s', form='eid'))
neg_eid_sample_dict['stress', 's_m', 'mi'] = hg.edge_type_subgraph(['s_m']).find_edges(
    hg.edges(etype='s_m', form='eid'))

neg_hetero_g = dgl.heterograph(neg_eid_sample_dict)
neg_hetero_g.ndata['h'] = hg.ndata['h']
neg_hetero_g = neg_hetero_g
all_neg_adj = data.get_adj(neg_hetero_g)
case_adj = np.ones_like(all_adj)
case_adj = case_adj - all_adj
node_feat_dict = {
    'lnc': lnc_feats,
    'mi': mi_feats,
    'stress': stress_feats
}
case_hg = get_case_graph()

node_dict = {}
edge_dict = {}
for ntype in hg.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in hg.etypes:
    edge_dict[etype] = len(edge_dict)

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

num_nodes = all_adj.shape[0]

# 模型定义参数
# in_feats, h_feats, out_feats,
# adj, Con_Us, p, q, walk_length, window_size
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    walk_length, window_size, n_layers, n_headers = 5, 5 * 0.4, 5, 2
    cv_case_pre_results = []

    Con_Us = cal_conUs(num_nodes, seed=3047 - 0)
    Con_Us_dict = {
        'lnc': Con_Us[:num_lnc], 'mi': Con_Us[num_lnc:num_lnc + num_mi], 'stress': Con_Us[num_lnc + num_mi:]}
    # 构建扰动后的X，并返回到hg中
    Xs = hg.ndata['h']
    perturbed_Xs = {k: torch.from_numpy(get_perturbation(Con_Us_dict[k], v)) for k, v in Xs.items()}
    perturbed_hg = copy.deepcopy(case_hg)
    perturbed_hg.ndata['h'] = perturbed_Xs
    model = CW_GNN(in_dims, h_dim, out_dim, all_adj, Con_Us, p, q, walk_length, window_size,
                   node_dict, edge_dict, n_layers=n_layers, n_headers=n_headers,
                   multi_n=args.NumMultiWalk, is_hetero=args.isHetero).to(device)
    opt = Ranger(model.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(total=num_epochs, dynamic_ncols=False)
    for epoch in range(100):
        pos_score, neg_score, causal_walks_dict = model(all_adj, all_neg_adj, perturbed_Xs, hg.to(device),
                                                        neg_hetero_g.to(device), ('lnc', 'l_s', 'stress'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.update(1)
        pbar.set_description(f'epoch:{epoch},loss:{loss.item():.3f}')
    # model.load_state_dict(torch.load(f'./定量分析/15_128_128_5_5_2_HGT_{i}.pth'))
    torch.save(model.state_dict(), f'./定量分析/all_trained_model.pth')
    model.eval()
    model = model.to(device)
    pos_score, _, _ = model(case_adj, case_adj, perturbed_Xs, case_hg,
                            case_hg, ('lnc', 'l_s', 'stress'))
    cv_case_pre_results.append(pos_score)
    for i in range(len(cv_case_pre_results)):
        cv_case_pre_results[i] = torch.sigmoid(cv_case_pre_results[i]).detach().cpu().numpy()
    l_s_row, l_s_col = np.nonzero(case_adj[:num_lnc, num_lnc + num_mi:])
    cv_case_pre_results = np.array(cv_case_pre_results)
    cv_case_pre_results = cv_case_pre_results.reshape(8498, 1)
    mean_cv_case_pre_results = np.mean(cv_case_pre_results, axis=1).reshape(-1, 1)
    case_final_results = np.concatenate([l_s_row.reshape(-1, 1), l_s_col.reshape(-1, 1), mean_cv_case_pre_results],
                                        axis=1)
    sorted_array = case_final_results[case_final_results[:, 2].argsort()[::-1]]
    pd.DataFrame(sorted_array).to_csv('./案例分析/case_final_results1.csv', index=False, header=False)
