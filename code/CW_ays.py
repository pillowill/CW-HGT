import json
import numpy as np
from sklearn.metrics import auc, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score, f1_score
from bulit_graphs import lnc_mi_stress_graphs
import CausalWalk as CW
import utils
import os


def cal_metrics(true_matrix, pred_matrix, k=10):
    # 将矩阵展平成一维数组
    pred_flat = pred_matrix.flatten()
    true_flat = true_matrix.flatten()

    # 获取正样本和负样本的索引
    positive_indices = np.where(true_flat == 1)[0]
    negative_indices = np.where(true_flat == 0)[0]

    # 随机选择相同数量的正负样本
    num_samples = min(len(positive_indices), len(negative_indices))
    selected_positive_indices = np.random.choice(positive_indices, num_samples, replace=False)
    selected_negative_indices = np.random.choice(negative_indices, num_samples, replace=False)

    # 合并正负样本索引
    selected_indices = np.concatenate((selected_positive_indices, selected_negative_indices))
    np.random.shuffle(selected_indices)

    # 根据选中的索引获取预测值和真实值
    selected_pred = pred_flat[selected_indices]
    selected_true = true_flat[selected_indices]

    # 计算标准二分类评价指标
    accuracy = accuracy_score(selected_true, selected_pred)
    recall = recall_score(selected_true, selected_pred)
    precision1 = precision_score(selected_true, selected_pred)
    f1 = f1_score(selected_true, selected_pred)

    # 计算 AP（Average Precision）
    ap_score = average_precision_score(selected_true, selected_pred)

    # 计算 Precision@k, Recall@k, Hits@k
    sorted_indices = np.argsort(-selected_pred)  # 从大到小排序
    top_k_indices = sorted_indices[:k]

    precision_at_k = np.sum(selected_true[top_k_indices]) / k
    recall_at_k = np.sum(selected_true[top_k_indices]) / np.sum(selected_true)
    hits_at_k = np.sum(selected_true[top_k_indices])

    # 计算 Specificity
    TP = np.sum((selected_pred == 1) & (selected_true == 1))
    FP = np.sum((selected_pred == 1) & (selected_true == 0))
    TN = np.sum((selected_pred == 0) & (selected_true == 0))
    specificity = TN / (TN + FP)

    return [precision1,recall,accuracy,f1,ap_score, precision_at_k,recall_at_k,hits_at_k,specificity]


def read_json2arr(json_file):
    # 读取JSON文件
    with open(json_file, 'r') as file:
        adjacency_list = json.load(file)

    # 获取所有节点
    nodes = list(adjacency_list.keys())
    node_count = len(nodes)

    # 创建节点到索引的映射
    node_index = {int(node): index for index, node in enumerate(nodes)}

    # 初始化邻接矩阵
    adjacency_matrix = np.zeros((node_count, node_count), dtype=int)

    # 填充邻接矩阵
    for node, neighbors in adjacency_list.items():
        if isinstance(neighbors[0], list):
            neighbors = neighbors[0]
        for neighbor in neighbors:
            if neighbor != int(node):  # 排除自环的情况
                i = node_index[int(node)]
                j = node_index[neighbor]
                adjacency_matrix[i][j] = 1
    return adjacency_matrix


def get_walks(adj, Con_Us, walk_type, p, q, walk_length, window_size, multi_n):
    if walk_type == 'CW':
        causal_walker = CW.MultiCausalWalk(p, q, walk_length, window_size, multi_n)
        causal_walks_list, causal_walks_dict = causal_walker(adj, Con_Us)
    elif walk_type == 'RW':
        n_nodes = adj.shape[0]
        causal_walks_list, causal_walks_dict = CW.RandomWalk(adj, n_nodes, walk_length).random_walks()
    else:
        n_nodes = adj.shape[0]
        causal_walks_list, causal_walks_dict = CW.N2V_RandomWalk(adj, n_nodes, p, q,
                                                                 walk_length).node2vec_random_walks()
    return causal_walks_list, causal_walks_dict


if __name__ == '__main__':
    walk_type = 'N2VRW'
    walk_length = 50
    window_size = int(0.4 * walk_length)
    n_lnc = 175
    n_mi = 500
    w_save_path = f'./定量分析/{walk_type}/'

    data = lnc_mi_stress_graphs.MYdataset()
    hg = data[0]
    lms_arr = data.get_adj(hg)
    ls_true_arr = data.get_adj(hg)[:n_lnc, n_lnc + n_mi:]
    results = []
    for i in range(10):
        if os.path.exists(f'./定量分析/{walk_type}/{i}_saved_{walk_length}_walks.json'):
            w_arr = read_json2arr(f'./定量分析/{walk_type}/{i}_saved_{walk_length}_walks.json')
        else:
            Con_Us = utils.cal_conUs(lms_arr.shape[0], is_test=True)
            w_list, w_dict = get_walks(lms_arr, Con_Us, walk_type, 3, 4, walk_length, window_size, 1)
            utils.save_causal_walks(w_save_path, w_dict, walk_length, i)
            w_arr = read_json2arr(f'./定量分析/{walk_type}/{i}_saved_{walk_length}_walks.json')

        ls_w_arr = w_arr[:n_lnc, n_lnc + n_mi:]
        results.append(cal_metrics(ls_true_arr, ls_w_arr))
    ave_results = np.mean(results, axis=0)
    std_results = np.std(results, axis=0)

    # CW acc:0.7063
    # RW acc:0.7063
