# c_utils.pyx

import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cal_conUs(int num_nodes):
    return np.random.randint(1, 3, num_nodes)

@cython.boundscheck(False)
@cython.wraparound(False)
def rand_cat_fast(np.ndarray[np.float64_t, ndim=1] p, int N):
    cdef int K = len(p)
    cdef np.ndarray[np.float64_t, ndim=2] u = np.random.rand(N, 1)
    cdef np.ndarray[np.float64_t, ndim=1] P = np.cumsum(p, axis=0)
    cdef np.ndarray[np.float64_t, ndim=2] U = np.tile(P.T, (N, 1))
    cdef np.ndarray[np.float64_t, ndim=2] c = np.tile(u, (1, K)) >= U
    c = c + 0
    cdef np.ndarray[np.int32_t, ndim=1] x = np.sum(c, 1) + 1
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
def histcnd(np.ndarray[np.int32_t, ndim=1] Y, np.ndarray[np.int32_t, ndim=1] U,
            np.ndarray[np.int32_t, ndim=1] yr, np.ndarray[np.int32_t, ndim=1] ur,
            int classes):
    cdef np.ndarray[np.int32_t, ndim=2] Nyu = np.zeros((classes, 2), dtype=np.int32)
    cdef int index, i
    for index, i in enumerate(Y):
        Nyu[i - 1][U[index] - 1] += 1
    return Nyu

@cython.boundscheck(False)
@cython.wraparound(False)
def cb_backdoor(list X,
                np.ndarray[np.int32_t, ndim=1] Y,
                list U,
                np.ndarray[np.int32_t, ndim=1] yr,
                np.ndarray[np.int32_t, ndim=1] ur,
                int M,
                int classes):
    cdef int N = len(X)
    cdef int K = yr.shape[0]
    cdef np.ndarray[np.int32_t, ndim=2] Nyu = histcnd(Y, np.array(U, dtype=np.int32), yr, ur, classes)
    cdef np.ndarray[np.float64_t, ndim=2] pyu = Nyu / N
    cdef np.ndarray[np.float64_t, ndim=2] pu = np.sum(pyu, axis=0, keepdims=True).T
    cdef np.ndarray[np.float64_t, ndim=2] py = np.sum(pyu, axis=1, keepdims=True)
    cdef np.ndarray[np.float64_t, ndim=2] py_u = pyu / pu.T
    cdef np.ndarray[np.int32_t, ndim=1] Ns = np.sum(Nyu, axis=1)
    cdef np.ndarray[np.float64_t, ndim=2] v = np.zeros((M, 1), dtype=np.float64)
    cdef int i, index, tmp
    cdef int j
    results_dict = {}
    for tmp in prange(Y.shape[0], nogil=True):
        i = Y[tmp]
        for index in range(Y.shape[0]):
            j = Y[index]
            with gil:
                if j == i:
                    v[index][0] = j / (py_u[i - 1][U[index] - 1]).T / N
    # for i in Y:
    #     for index, j in enumerate(Y):
    #         if j == i:
    #             v[index][0] = j / (py_u[i - 1][U[index] - 1]).T / N
    for i in prange(N, nogil=True):
        with gil:
            results_dict[X[i]] = v[i]
    return results_dict

@cython.boundscheck(False)
@cython.wraparound(False)
def get_perturbation(np.ndarray[np.int32_t, ndim=1] con_Us, np.ndarray[np.int32_t, ndim=2] X):
    cdef int i, j
    cdef np.ndarray[np.int32_t, ndim=2] x_gauss = np.zeros_like(X)
    cdef np.ndarray[np.int32_t, ndim=1] obs
    cdef np.ndarray[np.int32_t, ndim=2] obs_x = np.zeros((1, X.shape[1]), dtype=np.int32)

    # 获取 GIL 并生成随机数

    obs = np.random.binomial(1, 0.7, X.shape[1])

    for i in prange(con_Us.shape[0], nogil=True):
        if con_Us[i] == 1:
            for j in range(X.shape[1]):
                if obs[j] == 1:
                    obs_x[0, j] = X[i, j]
                else:
                    if X[i, j] == 1:
                        obs_x[0, j] = 0
                    else:
                        obs_x[0, j] = 1
            for j in range(X.shape[1]):  # copying data back
                x_gauss[i, j] = obs_x[0, j]
        elif con_Us[i] == 2:
            for j in range(X.shape[1]):
                x_gauss[i, j] = X[i, j]
    return x_gauss

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_transition_probs(dict neighbor_table, float p, float q, dict w, object prev_node=None):
    cdef int idx, length
    cdef list probs = []
    cdef list neighbors = []
    cdef float causal_w
    length = len(w)
    neighbors = list(w.keys())
    for idx in prange(length, nogil=True):
        with gil:
            neighbor = neighbors[idx]
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
    # for neighbor, causal_w in w.items():
    #     causal_w = w[neighbor]
    #     if prev_node is None:
    #         probs.append((1.0 * causal_w) + 1)
    #     else:
    #         if neighbor == prev_node:
    #             probs.append((1.0 / p) + (1.0 / p) * causal_w)
    #         elif neighbor in neighbor_table[prev_node]:
    #             probs.append(1 + (1.0 * causal_w))
    #         else:
    #             probs.append((1.0 / q) + (1.0 / q) * causal_w)

    return np.array(probs) / np.sum(np.array(probs))

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def generate_context_pairs(list walks, int window_size):
#     cdef list pairs = []
#     cdef int i, j
#     cdef int word
#     for walk in walks:
#         for i in range(len(walk)):
#             word = walk[i]
#             for j in prange(max(0, i - window_size), min(len(walk), i + window_size + 1), nogil=True):
#                 if i != j:
#                     pairs.append((int(word), int(walk[j])))
#     return np.array(pairs)
