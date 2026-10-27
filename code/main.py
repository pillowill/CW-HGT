# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/5 17:00
@Auth ： He Yu
@File ：main.py
@IDE ：PyCharm
@Function ：Function of the script
训练模型的逻辑
1.数据准备：加载数据并进行负采样，对正图和负图进行5折划分
2.数据处理：对于每个训练图得到社区划分Y
          使用causal walk进行（多重）采样
3.再使用GAT对（多重）采样进行学习和聚合得到每个节点的嵌入
"""
import utils
import set_params
from utils import *
from tqdm.contrib import tzip
from bulit_graphs import lnc_mi_stress_graphs
from bulit_graphs import lnc_stress_graphs
from bulit_graphs import mi_stress_graphs
# from bulit_graphs.ab_graphs import lnc_stress_graphs, lnc_mi_stress_graphs
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from ranger import Ranger
import torch
import copy
from CW_GNN import CW_GNN, ResNet
import dgl
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, auc, confusion_matrix

device = torch.device('cuda')


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
    TP = cm[0, 0]
    FP = cm[1, 0]
    FN = cm[0, 1]
    TN = cm[1, 1]
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


def train(in_dims, h_dim, out_dim, p=1, q=1, walk_length=10, window_size=5, num_epoch=100, lr=0.001,
          weight_decay=0.001, k_fold=5):
    # 进行5折交叉
    # 每一折交叉里面使用pos g学习节点特征，使用neg g获得负样本预测值
    cv_models = []
    for i in range(k_fold):
        set_random_seed(3047 - i)
        tr_hg = pos_tr_gs[i]
        tr_neg_hg = neg_tr_gs[i]
        adj = data.get_adj(tr_hg)
        neg_adj = data.get_adj(tr_neg_hg)
        num_nodes = adj.shape[0]
        Con_Us = cal_conUs(num_nodes, 3047 - i)
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
                       node_dict, edge_dict, multi_n=args.NumMultiWalk,
                       is_hetero=args.isHetero).to(device)
        savePath = f'{args.WalkSavePath}/train'
        causal_walks_dict = model.return_walk_dict()
        if causal_walks_dict is not None:
            utils.save_causal_walks(savePath, causal_walks_dict, walk_length, i)
        opt = Ranger(model.parameters(), lr=lr, weight_decay=weight_decay)
        pbar = tqdm(total=num_epochs, dynamic_ncols=False)
        for epoch in range(num_epoch):
            if args.DataClass == 'lnc_mi_stress_graphs':
                if pre_edge_type == 'l_s':
                    pos_score, neg_score, causal_walks_dict, meta_attn_scores = \
                        model(adj, neg_adj, perturbed_Xs, tr_hg, tr_neg_hg,
                              ('lnc', 'l_s', 'stress'))
                else:
                    pos_score, neg_score, causal_walks_dict, meta_attn_scores = \
                        model(adj, neg_adj, perturbed_Xs, tr_hg, tr_neg_hg,
                              ('mi', 'm_s', 'stress'))
            elif args.DataClass == 'lnc_stress_graphs':
                pos_score, neg_score, causal_walks_dict, meta_attn_scores = \
                    model(adj, neg_adj, perturbed_Xs, tr_hg, tr_neg_hg,
                          ('lnc', 'l_s', 'stress'))
            else:
                pos_score, neg_score, causal_walks_dict, meta_attn_scores = \
                    model(adj, neg_adj, perturbed_Xs, tr_hg, tr_neg_hg,
                          ('mi', 'm_s', 'stress'))

            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.update(1)
            pbar.set_description(f'cv:{i},epoch:{epoch},loss:{loss.item():.3f}')
            # Clear cached memory
            del loss, pos_score, neg_score
            torch.cuda.empty_cache()
        cv_models.append(model)
        pbar.close()
    return cv_models


def test(pos_g, neg_g, model_save_path, in_dims, h_dim, out_dim, idx, p=1, q=1, walk_length=10,
         window_size=5):
    te_hg = pos_g
    te_neg_hg = neg_g
    adj = data.get_adj(te_hg)
    neg_adj = data.get_adj(te_neg_hg)
    num_nodes = adj.shape[0]
    Con_Us = cal_conUs(num_nodes, is_test=True)
    node_feats = hg.ndata['h']
    model = CW_GNN(in_dims, h_dim, out_dim, adj, Con_Us, p, q, walk_length, window_size,
                   node_dict,
                   edge_dict,
                   is_hetero=args.isHetero).to(device)
    model.load_state_dict(torch.load(f'{model_save_path}'))
    model.eval()
    with torch.no_grad():
        if args.DataClass == 'lnc_mi_stress_graphs':
            if pre_edge_type == 'l_s':
                pos_score, neg_score, causal_walks_dict, meta_attn_scores = \
                    model(adj, neg_adj, node_feats, te_hg, te_neg_hg,
                          ('lnc', 'l_s', 'stress'))
            else:
                pos_score, neg_score, causal_walks_dict, meta_attn_scores = \
                    model(adj, neg_adj, node_feats, te_hg, te_neg_hg,
                          ('mi', 'm_s', 'stress'))
        elif args.DataClass == 'lnc_stress_graphs':
            pos_score, neg_score, causal_walks_dict, meta_attn_scores = \
                model(adj, neg_adj, node_feats, te_hg, te_neg_hg,
                      ('lnc', 'l_s', 'stress'))
        else:
            pos_score, neg_score, causal_walks_dict, meta_attn_scores = \
                model(adj, neg_adj, node_feats, te_hg, te_neg_hg,
                      ('mi', 'm_s', 'stress'))
        savePath = f'{args.WalkSavePath}/test'
        if causal_walks_dict is not None:
            utils.save_causal_walks(savePath, causal_walks_dict, walk_length, idx)
        # pos_score, neg_score = model(adj, neg_adj, node_feats, te_hg, te_neg_hg, ('mi', 'm_s', 'stress'))
    return pos_score, neg_score, meta_attn_scores


if __name__ == '__main__':
    # 获取所有的参数
    args = set_params.get_args()
    num_lnc, num_mi, num_stress = args.num_lnc, args.num_mi, args.num_stress
    DataClass = args.DataClass
    if DataClass == 'lnc_mi_stress_graphs':
        data = lnc_mi_stress_graphs.MYdataset()
        hg = data[0]
        num_lnc = hg.num_nodes('lnc')
        num_mi = hg.num_nodes('mi')
        num_stress = hg.num_nodes('stress')
        node_feats = hg.ndata['h']
        lnc_feats = node_feats['lnc']
        mi_feats = node_feats['mi']
        stress_feats = node_feats['stress']
        in_dims = [lnc_feats.shape[1], mi_feats.shape[1], stress_feats.shape[1]]
    elif DataClass == 'lnc_stress_graphs':
        data = lnc_stress_graphs.MYdataset()
        hg = data[0]
        num_lnc = hg.num_nodes('lnc')
        num_stress = hg.num_nodes('stress')
        node_feats = hg.ndata['h']
        lnc_feats = node_feats['lnc']
        stress_feats = node_feats['stress']
        in_dims = [lnc_feats.shape[1], stress_feats.shape[1]]
    else:
        data = mi_stress_graphs.MYdataset()
        hg = data[0]
        num_mi = hg.num_nodes('mi')
        num_stress = hg.num_nodes('stress')
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

    pre_edge_type = args.pre_edge_type
    kFold = args.kFold
    k_kNN = args.kNN
    pos_tr_gs, pos_te_gs = data.split_train_test(hg, kFold, pre_edge_type)
    neg_tr_gs, neg_te_gs = data.negative_sample(hg, k=1, kFold=kFold, pre_edge=pre_edge_type)

    k_fold = args.kFold
    knn_k = args.kNN
    h_dim = args.HidDim
    out_dim = args.OutDim
    n_layers = args.nLayers
    n_headers = args.nHeaders
    p = args.p
    q = args.q
    walk_length = args.WalkLength
    window_size = int(0.4 * walk_length)
    num_epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    trained_cv_models = []
    if args.isHetero:
        model_save_path = f'{args.ModelSavePath}/' \
                          f'{knn_k}_{h_dim}_{out_dim}_{walk_length}_{n_layers}_{n_headers}_HGT'
    else:
        model_save_path = f'{args.ModelSavePath}/' \
                          f'{knn_k}_{h_dim}_{out_dim}_{walk_length}_{n_layers}_{n_headers}_GCN'

    if ~os.path.exists(f'{args.ModelSavePath}/{knn_k}_{h_dim}_{out_dim}_{walk_length}_{n_layers}_{n_headers}_HGT_0'):
        trained_cv_models = train(in_dims, h_dim, out_dim, p, q, walk_length,
                                  window_size, num_epochs, lr=lr,
                                  weight_decay=weight_decay, k_fold=k_fold)
        test_metrics = []
        fold_test_meta_attn_arr = []
        for i in range(len(trained_cv_models)):
            torch.save(trained_cv_models[i].state_dict(), f'{model_save_path}_{i}.pth')
        test_metrics = []
        fold_test_meta_attn_arr = []
        for i in range(k_fold):
            set_random_seed(3047 - i)
            # project_path = f'{save_path1}_{i}.pth'
            model_path = f'{model_save_path}_{i}.pth'
            te_hg = pos_te_gs[i]
            te_neg_hg = neg_te_gs[i]
            idx = i
            pos_score, neg_score, meta_attn_scores = test(te_hg, te_neg_hg, model_path, in_dims, h_dim, out_dim, i,
                                                          p, q,
                                                          walk_length,
                                                          window_size,
                                                          )
            if meta_attn_scores is not None:
                meta_attn_arr = utils.attn_arr(meta_attn_scores, n_layers)
                fold_test_meta_attn_arr.append(meta_attn_arr)
            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)
            labels = torch.cat([pos_labels, neg_labels])
            scores = torch.cat([pos_score, neg_score])
            result = calc_metrics(scores, labels)
            test_metrics.append(result)

        if len(fold_test_meta_attn_arr) > 0:
            fold_test_meta_attn_dict = {i: np.zeros(shape=(3, 3)) for i in range(n_layers)}
            for l in range(n_layers):
                for arr in fold_test_meta_attn_arr:
                    fold_test_meta_attn_dict[l] += arr[l]
                fold_test_meta_attn_dict[l] = fold_test_meta_attn_dict[l] / k_fold
            with open(f'./参数设置/{knn_k}_{h_dim}_{out_dim}_{walk_length}_{n_layers}_{n_headers}_meta_attn_arr.json',
                      "w+",
                      encoding='utf-8') as f:
                f.write(json.dumps(fold_test_meta_attn_dict, ensure_ascii=False, indent=4, cls=NumpyEncoder))
        ave_test_metrics = np.array(test_metrics[:7])
        ave_test_metrics = np.mean(ave_test_metrics, axis=0)
        std_test_metrics = np.std(test_metrics, axis=0)
    else:
        test_metrics = []
        fold_test_meta_attn_arr = []
        for i in range(k_fold):
            set_random_seed(3047 - i)
            # project_path = f'{save_path1}_{i}.pth'
            model_path = f'{model_save_path}_{i}.pth'
            te_hg = pos_te_gs[i]
            te_neg_hg = neg_te_gs[i]
            idx = i
            pos_score, neg_score, meta_attn_scores = test(te_hg, te_neg_hg, model_path, in_dims, h_dim, out_dim, i, p, q,
                                                          walk_length,
                                                          window_size,
                                                          )
            if meta_attn_scores is not None:
                meta_attn_arr = utils.attn_arr(meta_attn_scores, n_layers)
                fold_test_meta_attn_arr.append(meta_attn_arr)
            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)
            labels = torch.cat([pos_labels, neg_labels])
            scores = torch.cat([pos_score, neg_score])
            result = calc_metrics(scores, labels)
            test_metrics.append(result)

        if len(fold_test_meta_attn_arr) > 0:
            fold_test_meta_attn_dict = {i: np.zeros(shape=(3, 3)) for i in range(n_layers)}
            for l in range(n_layers):
                for arr in fold_test_meta_attn_arr:
                    fold_test_meta_attn_dict[l] += arr[l]
                fold_test_meta_attn_dict[l] = fold_test_meta_attn_dict[l] / k_fold
            with open(f'./参数设置/{knn_k}_{h_dim}_{out_dim}_{walk_length}_{n_layers}_{n_headers}_meta_attn_arr.json', "w+",
                      encoding='utf-8') as f:
                f.write(json.dumps(fold_test_meta_attn_dict, ensure_ascii=False, indent=4, cls=NumpyEncoder))
        ave_test_metrics = np.array(test_metrics[:7])
        ave_test_metrics = np.mean(ave_test_metrics, axis=0)
        std_test_metrics = np.std(test_metrics, axis=0)
