# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/5 16:49
@Auth ： He Yu
@File ：utils.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import os.path
import numpy as np
import json


def cal_conUs(num_nodes, seed):
    np.random.seed(seed)
    return np.random.randint(1, 3, num_nodes)


def rand_cat_fast(p, N):
    K = len(p)
    u = np.random.rand(N, 1)
    P = np.cumsum(p, axis=0)
    U = np.tile(P.T, (N, 1))
    c = np.tile(u, (1, K)) >= U
    c = c + 0
    x = np.sum(c, 1) + 1
    return x


def histcnd(Y, U, yr, ur, classes):
    Nyu = np.zeros((classes, 2), dtype=int)
    for index, i in enumerate(Y):
        Nyu[i - 1][U[index] - 1] += 1

    return Nyu


def cb_backdoor(X, Y, U, yr, ur, M, classes):
    # X的数据类型维list， U的数据类型为list，
    N = len(X)
    K = yr.shape[0]
    Nyu = histcnd(Y, U, yr, ur, classes)  # Y和U的联合频数矩阵
    pyu = Nyu / N  # Y和U的联合概率分布
    pu = np.sum(pyu, axis=0, keepdims=True).T  # U的边缘概率分布
    py = np.sum(pyu, axis=1, keepdims=True)  # Y的边缘概率分布
    py_u = pyu / pu.T  # 在Y条件下U的条件概率分布
    Ns = np.sum(Nyu, axis=1)

    v = np.zeros((M, 1), dtype=float)

    for i in Y:
        for index, j in enumerate(Y):
            if j == i:
                v[index][0] = j / (py_u[i - 1][U[index] - 1]).T / N
    results_dict = {}
    for i in range(N):
        results_dict[X[i]] = v[i]
    return results_dict


def get_perturbation(con_Us, X):
    x_gauss = np.zeros_like(X)
    obs = np.random.binomial(1, 0.7, X.shape[1])
    obs_x = np.zeros((1, X.shape[1]))
    for i, val in enumerate(con_Us):
        if con_Us[i] == 1:
            for j, val1 in enumerate(X[i, :]):
                if obs[j] == 1:
                    obs_x[0][j] = val1
                elif obs[j] == 0:
                    if val1 == 1:
                        obs_x[0][j] = 0
                    elif val1 == 0:
                        obs_x[0][j] = 1
            x_gauss[i] = obs_x
        elif con_Us[i] == 2:
            x_gauss[i] = X[i, :]
    # xc = x_gauss / x_gauss.sum(1, keepdims=True)  # Normalize the data
    return x_gauss


def compute_transition_probs(neighbor_table, p, q, w, prev_node=None):
    # neighbor_table的数据类型为字典，
    probs = []
    for neighbor, causal_w in w.items():
        causal_w = w[neighbor]
        if prev_node is None:
            probs.append((1.0 * causal_w) + 1)
        else:
            if neighbor == prev_node:
                probs.append((1.0 / p) + (1.0 / p) * causal_w)
            elif neighbor in neighbor_table[prev_node]:
                probs.append(1 + (1.0 * causal_w))
            else:
                probs.append((1.0 / q) + (1.0 / q) * causal_w)

    probs = np.array(probs)
    return probs / np.sum(probs)


def generate_context_pairs(walks, window_size):
    pairs = []
    for walk in walks:
        for i, word in enumerate(walk):
            for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                if i != j:
                    pairs.append((int(word), int(walk[j])))
    return np.array(pairs)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_causal_walks(savePath, walk_dict, walk_len, idx):
    with open(f"{savePath}/{idx}_saved_{walk_len}_walks.json", "w+", encoding='utf-8') as f:
        f.write(json.dumps(walk_dict, ensure_ascii=False, indent=4, cls=NumpyEncoder))


def load_cache_walks(savePath, walk_satus, walk_len, idx):
    if os.path.exists(savePath):
        pass


def check_duplicate_rows(array):
    # Convert the array to a structured array with a single compound data type
    structured_array = np.ascontiguousarray(array).view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))

    # Find duplicates by checking unique values
    _, unique_indices = np.unique(structured_array, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(array.shape[0]), unique_indices)

    if len(duplicate_indices) > 0:
        return duplicate_indices
    else:
        print("No duplicate rows found.")
        return False


def attn_arr(meta_attn_scores, n_layers):
    meta_attn_arrs = {}
    for l in range(n_layers):
        meta_attn_score = [np.max(i.cpu().numpy()) for i in meta_attn_scores[l]]
        meta_attn_arr = np.zeros(shape=(3, 3))
        meta_attn_arr[0, 0] = meta_attn_score[0]
        meta_attn_arr[0, 1] = meta_attn_score[2]
        meta_attn_arr[0, 2] = meta_attn_score[1]
        meta_attn_arr[1, 0] = meta_attn_score[6]
        meta_attn_arr[1, 1] = meta_attn_score[4]
        meta_attn_arr[1, 2] = meta_attn_score[5]
        meta_attn_arr[2, 0] = meta_attn_score[3]
        meta_attn_arr[2, 1] = meta_attn_score[7]
        meta_attn_arr[2, 2] = meta_attn_score[8]
        meta_attn_arrs[l] = meta_attn_arr
    return meta_attn_arrs


def read_causal_walks(file_path):
    with open(file_path, 'r') as file:
        pass


if __name__ == '__main__':
    pass
