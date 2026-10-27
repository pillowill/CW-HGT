# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/26 15:11
@Auth ： He Yu
@File ：draw_auc.py
@IDE ：PyCharm
@Function ：Function of the script
"""
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@Time ： 2024/10/30 10:56
@Auth ： He Yu
@File ：new_main.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import utils
import set_params
from utils import *
import scipy.io
from torch.optim import Adam
from tqdm.contrib import tzip
import pickle
from bulit_graphs import lnc_mi_stress_graphs
from bulit_graphs import lnc_stress_graphs
from bulit_graphs import mi_stress_graphs
import torch
# from bulit_graphs.ab_graphs import lnc_stress_graphs, lnc_mi_stress_graphs
import torch.nn.functional as F
from tqdm import tqdm
from ranger import Ranger
import copy
import dgl
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, auc, confusion_matrix
from CW_GNN import CW_GNN

# from grid_search_paramters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calc_metrics(scores, labels):
    preds = (scores > torch.mean(torch.sigmoid(scores))).float()
    # preds = (scores > 0.5).float()
    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    prec = precision_score(labels.cpu().numpy(), preds.cpu().numpy())
    rec = recall_score(labels.cpu().numpy(), preds.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
    cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    my_auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    aupr = average_precision_score(labels.cpu().numpy(), scores.cpu().numpy())
    # 修正混淆矩阵索引
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    specificity = TN / (TN + FP)
    return [my_auc, aupr, prec, rec, acc, f1, specificity]


def compute_loss(pos_score, neg_score):
    pos_labels = torch.ones_like(pos_score)
    neg_labels = torch.zeros_like(neg_score)
    labels = torch.cat([pos_labels, neg_labels])
    scores = torch.cat([pos_score, neg_score])
    # 与标签的损失函数
    loss1 = F.binary_cross_entropy(torch.sigmoid(scores), labels)
    # # 间隔损失
    n_edges = pos_score.shape[0]
    loss2 = (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()
    return (loss1 + loss2) / 2
    # return loss1


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


def cv_train(pos_tr_gs, pos_te_gs, neg_tr_gs, neg_te_gs,
             in_dims, h_dim, out_dim, p=1, q=1, walk_length=10, window_size=5, n_layers=4, n_headers=8,
             num_epoch=100, lr=0.001, weight_decay=0.001, k_fold=5):
    cv_models = []
    test_metrics = []
    test_results = []
    for i in range(k_fold):
        set_random_seed(3047 - i)
        tr_hg = pos_tr_gs[i]
        tr_neg_hg = neg_tr_gs[i]
        te_hg = pos_te_gs[i]
        te_neg_hg = neg_te_gs[i]
        adj = data.get_adj(tr_hg)
        neg_adj = data.get_adj(tr_neg_hg)
        num_nodes = adj.shape[0]
        Con_Us = cal_conUs(num_nodes, seed=3047 - i)
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
        # 模型定义参数in_feats, h_feats, out_feats, adj, Con_Us, p, q, walk_length, window_size
        model = CW_GNN(in_dims, h_dim, out_dim, adj, Con_Us, p, q, walk_length, window_size,
                       node_dict, edge_dict, n_layers=n_layers, n_headers=n_headers,
                       multi_n=args.NumMultiWalk, is_hetero=args.isHetero).to(device)
        model.load_state_dict(torch.load(
            f'trained_models/lnc_mi_stress_graphs/l_s/CW/single_path/outer_path/15_128_128_5_5_2_HGT_{i}.pth'))
        model.eval()
        with torch.no_grad():
            te_pos_score, te_neg_score, causal_walks_dict = model(adj, neg_adj, node_feats, te_hg,
                                                                  te_neg_hg, ('lnc', 'l_s', 'stress'))
        te_pos_labels = torch.ones_like(te_pos_score)
        te_neg_labels = torch.zeros_like(te_neg_score)
        te_labels = torch.cat([te_pos_labels, te_neg_labels])
        te_scores = torch.cat([te_pos_score, te_neg_score])
        test_results.append([te_scores.cpu().numpy(), te_labels.cpu().numpy()])
        te_result = calc_metrics(te_scores, te_labels)
        print(te_result)
        test_metrics.append(te_result)
    ave_test_metrics = np.mean(test_metrics, axis=0)
    # print(ave_test_metrics)
    std_test_metrics = np.std(test_metrics, axis=0)
    return cv_models, ave_test_metrics, std_test_metrics, test_results


if __name__ == '__main__':
    # 获取所有的参数
    args = set_params.get_args()
    num_lnc, num_mi, num_stress = args.num_lnc, args.num_mi, args.num_stress
    DataClass = args.DataClass
    if DataClass == 'lnc_mi_stress_graphs':
        data = lnc_mi_stress_graphs.MYdataset()
        hg = data[0]
        node_feats = hg.ndata['h']
        lnc_feats = node_feats['lnc']
        mi_feats = node_feats['mi']
        stress_feats = node_feats['stress']
        in_dims = [lnc_feats.shape[1], mi_feats.shape[1], stress_feats.shape[1]]
    elif DataClass == 'lnc_stress_graphs':
        data = lnc_stress_graphs.MYdataset()
        hg = data[0]
        node_feats = hg.ndata['h']
        lnc_feats = node_feats['lnc']
        stress_feats = node_feats['stress']
        in_dims = [lnc_feats.shape[1], stress_feats.shape[1]]
    else:
        data = mi_stress_graphs.MYdataset()
        hg = data[0]
        node_feats = hg.ndata['h']
        mi_feats = node_feats['mi']
        stress_feats = node_feats['stress']
        in_dims = [mi_feats.shape[1], stress_feats.shape[1]]

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

    # walk_length_list = [5, 10, 15, 20, 25, 30, 35, 40]
    walk_length_list = [5]
    # HGT_layers_list = [1, 2, 3, 4, 5, 6]
    HGT_layers_list = [5]
    # headers_list = [1, 2, 4, 8, 16, 32]
    headers_list = [2]

    for walk_length in tqdm(walk_length_list):
        for layers in HGT_layers_list:
            for header in headers_list:
                print(f'{KNN_K}_{walk_length}_{layers}_{header}')
                window_size = int(0.4 * walk_length)
                cv_models, ave_test_metrics, std_test_metrics, te_results = cv_train(pos_tr_gs, pos_te_gs, neg_tr_gs,
                                                                                     neg_te_gs,
                                                                                     in_dims, h_dim, out_dim, p, q,
                                                                                     walk_length,
                                                                                     window_size, layers, header,
                                                                                     num_epochs, lr, weight_decay,
                                                                                     k_fold)
                if args.isHetero:
                    model_save_path = f'{args.ModelSavePath}/' \
                                      f'{KNN_K}_{h_dim}_{out_dim}_{walk_length}_{layers}_{header}_HGT'
                else:
                    model_save_path = f'{args.ModelSavePath}/' \
                                      f'{KNN_K}_{h_dim}_{out_dim}_{walk_length}_{layers}_{header}_GCN'
                # for i in range(len(cv_models)):
                #     torch.save(cv_models[i].state_dict(), f'{model_save_path}_{i}.pth')
                del cv_models
                print(f'{KNN_K}_{walk_length}_{layers}_{header}:{ave_test_metrics}')
                params_ave_test_metrics_dict[f'{KNN_K}_{walk_length}_{layers}_{header}'] = ave_test_metrics
                params_std_test_metrics_dict[f'{KNN_K}_{walk_length}_{layers}_{header}'] = std_test_metrics
    # np.savez("./参数设置/cv_grid_search_ave.npz", **params_ave_test_metrics_dict)
    # np.savez("./参数设置/cv_grid_search_std.npz", **params_std_test_metrics_dict)
