import dgl
import numpy as np
import torch
import pandas as pd
from dgl.data import DGLDataset
from sknetwork.clustering import Louvain
from sklearn.preprocessing import StandardScaler
import sys
import os
import set_params
from dgl import knn_graph

# 获取当前文件的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到 Python 搜索路径
sys.path.append(parent_dir)


class MYdataset(DGLDataset):
    def __init__(self):
        super(MYdataset, self).__init__(name='mydata')
        self.hg = self.process()
        self.edge_types = self.hg.etypes
        self.node_types = self.hg.ntypes
        self.args = set_params.get_args()
        self.link_types = [
            ['lnc', 'l_s', 'stress'],
            ['stress', 's_l', 'lnc'],
            ['mi', 'm_s', 'stress'],
            ['stress', 's_m', 'mi'],
            ['lnc', 'l_m', 'mi'],
            ['mi', 'm_l', 'lnc'],
            ['lnc', 'l_l', 'lnc'],
            ['mi', 'm_m', 'mi'],
            ['stress', 's_s', 'stress']
        ]

    @staticmethod
    def get_node_neighbors(g, edge_type):
        # g represents the meta-graph in a hetero_g
        node_types = g.ntypes
        node_neighbors_dict = {}
        for node_type in node_types:
            nodes = g.nodes(node_type)
            neighbors_list = []
            for node in nodes:
                try:
                    successors = g.successors(node, etype=edge_type)
                except:
                    successors = torch.tensor([], dtype=torch.int64)
                mask = successors != node
                neighbors = successors[mask]
                neighbors = torch.cat([torch.tensor([node]), neighbors])
                neighbors_list.append(neighbors)
            node_neighbors_dict[node_type] = neighbors_list
        return node_neighbors_dict

    @staticmethod
    def get_community_labels(g, edge_type):
        # g represents the meta graph of hg
        node_types = g.ntypes
        adj = np.array(g.adj(etype=edge_type).to_dense())
        m, n = adj.shape
        louvain = Louvain()
        bi_adj = np.concatenate(
            (np.concatenate((np.zeros((m, m)), adj), axis=1),
             np.concatenate((adj.T, np.zeros((n, n))), axis=1)),
            axis=0
        )
        labels = louvain.fit_transform(bi_adj)
        m_labels, n_labels = torch.from_numpy(labels[:m]), torch.from_numpy(labels[m:])
        if len(node_types) < 2:
            g.nodes[node_types[0]].data[f'{edge_type}_y'] = m_labels
        else:
            try:
                g.nodes[node_types[0]].data[f'{edge_type}_y'] = m_labels
                g.nodes[node_types[1]].data[f'{edge_type}_y'] = n_labels
            except:
                g.nodes[node_types[1]].data[f'{edge_type}_y'] = m_labels
                g.nodes[node_types[0]].data[f'{edge_type}_y'] = n_labels
        return g

    @staticmethod
    def remove_duplicate_rows(array):
        # Convert the array to a structured array with a single compound data type
        structured_array = np.ascontiguousarray(array).view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))

        # Use np.unique to remove duplicate rows
        unique_structured_array = np.unique(structured_array)

        # Convert back to the original shape
        unique_array = unique_structured_array.view(array.dtype).reshape(-1, array.shape[1])

        return unique_array

    def split_train_test(self, hetero_g, kFold=5, pre_edge='l_s'):
        np.random.seed(self.args.random_seed)
        # Define the number of edges for each edge type
        num_lnc2stress = hetero_g.num_edges('l_s')
        num_mi2lnc = hetero_g.num_edges('m_l')
        num_mi2stress = hetero_g.num_edges('m_s')
        num_l_l = hetero_g.num_edges('l_l')
        num_m_m = hetero_g.num_edges('m_m')
        num_s_s = hetero_g.num_edges('s_s')
        train_gs = []
        test_gs = []
        # Create folds for each edge type
        num_folds = kFold
        lnc2stress = np.arange(num_lnc2stress)
        np.random.shuffle(lnc2stress)
        lnc2stress_folds = np.array_split(lnc2stress, num_folds)

        mi2stress = np.arange(num_mi2stress)
        np.random.shuffle(mi2stress)
        # mi2lnc_folds = np.array_split(np.arange(num_mi2lnc), num_folds)
        mi2stress_folds = np.array_split(np.arange(num_mi2stress), num_folds)

        for fold_idx in range(num_folds):
            # Generate masks for training and testing edges based on the current fold
            train_idx_lnc2stress = np.concatenate([lnc2stress_folds[i] for i in range(num_folds) if i != fold_idx])
            # train_idx_mi2lnc = np.concatenate([mi2lnc_folds[i] for i in range(num_folds) if i != fold_idx])
            train_idx_mi2stress = np.concatenate([mi2stress_folds[i] for i in range(num_folds) if i != fold_idx])

            train_mask_lnc2stress = torch.zeros(num_lnc2stress, dtype=torch.bool)
            train_mask_lnc2stress[train_idx_lnc2stress] = True

            # train_mask_mi2lnc = torch.zeros(num_mi2lnc, dtype=torch.bool)
            # train_mask_mi2lnc[train_idx_mi2lnc] = True

            train_mask_mi2stress = torch.zeros(num_mi2stress, dtype=torch.bool)
            train_mask_mi2stress[train_idx_mi2stress] = True

            test_mask_lnc2stress = ~train_mask_lnc2stress
            # test_mask_mi2lnc = ~train_mask_mi2lnc
            test_mask_mi2stress = ~train_mask_mi2stress

            # Add these masks as edge features to the graph
            hetero_g.edges[('lnc', 'l_s', 'stress')].data['train_mask_fold_' + str(fold_idx)] = train_mask_lnc2stress
            hetero_g.edges[('lnc', 'l_s', 'stress')].data['test_mask_fold_' + str(fold_idx)] = test_mask_lnc2stress

            # hetero_g.edges[('mi', 'm_l', 'lnc')].data['train_mask_fold_' + str(fold_idx)] = train_mask_mi2lnc
            # hetero_g.edges[('mi', 'm_l', 'lnc')].data['test_mask_fold_' + str(fold_idx)] = test_mask_mi2lnc

            hetero_g.edges[('mi', 'm_s', 'stress')].data['train_mask_fold_' + str(fold_idx)] = train_mask_mi2stress
            hetero_g.edges[('mi', 'm_s', 'stress')].data['test_mask_fold_' + str(fold_idx)] = test_mask_mi2stress

            if pre_edge == 'l_s':
                # Create training and testing subgraphs for the current fold
                train_edge_dict = {
                    ('lnc', 'l_s', 'stress'): train_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
                    ('stress', 's_l', 'lnc'): train_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
                    ('mi', 'm_s', 'stress'): np.arange(num_mi2stress),
                    ('stress', 's_m', 'mi'): np.arange(num_mi2stress),
                    ('lnc', 'l_m', 'mi'): np.arange(num_mi2lnc),
                    ('mi', 'm_l', 'lnc'): np.arange(num_mi2lnc),
                    ('lnc', 'l_l', 'lnc'): np.arange(num_l_l),
                    ('mi', 'm_m', 'mi'): np.arange(num_m_m),
                    ('stress', 's_s', 'stress'): np.arange(num_s_s),
                }

                test_edge_dict = {
                    ('lnc', 'l_s', 'stress'): test_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
                    ('stress', 's_l', 'lnc'): test_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
                    ('mi', 'm_s', 'stress'): np.arange(num_mi2stress),
                    ('stress', 's_m', 'mi'): np.arange(num_mi2stress),
                    ('lnc', 'l_m', 'mi'): np.arange(num_mi2lnc),
                    ('mi', 'm_l', 'lnc'): np.arange(num_mi2lnc),
                    ('lnc', 'l_l', 'lnc'): np.arange(num_l_l),
                    ('mi', 'm_m', 'mi'): np.arange(num_m_m),
                    ('stress', 's_s', 'stress'): np.arange(num_s_s),

                }
            else:
                train_edge_dict = {
                    ('lnc', 'l_s', 'stress'): np.arange(num_lnc2stress),
                    ('stress', 's_l', 'lnc'): np.arange(num_lnc2stress),
                    ('mi', 'm_s', 'stress'): train_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
                    ('stress', 's_m', 'mi'): train_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
                    ('lnc', 'l_m', 'mi'): np.arange(num_mi2lnc),
                    ('mi', 'm_l', 'lnc'): np.arange(num_mi2lnc),
                    ('lnc', 'l_l', 'lnc'): np.arange(num_l_l),
                    ('mi', 'm_m', 'mi'): np.arange(num_m_m),
                    ('stress', 's_s', 'stress'): np.arange(num_s_s),
                }

                test_edge_dict = {
                    ('lnc', 'l_s', 'stress'): np.arange(num_lnc2stress),
                    ('stress', 's_l', 'lnc'): np.arange(num_lnc2stress),
                    ('mi', 'm_s', 'stress'): test_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
                    ('stress', 's_m', 'mi'): test_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
                    ('lnc', 'l_m', 'mi'): np.arange(num_mi2lnc),
                    ('mi', 'm_l', 'lnc'): np.arange(num_mi2lnc),
                    ('lnc', 'l_l', 'lnc'): np.arange(num_l_l),
                    ('mi', 'm_m', 'mi'): np.arange(num_m_m),
                    ('stress', 's_s', 'stress'): np.arange(num_s_s),
                }

            train_graph = dgl.edge_subgraph(hetero_g, train_edge_dict, relabel_nodes=False)
            test_graph = dgl.edge_subgraph(hetero_g, test_edge_dict, relabel_nodes=False)
            train_gs.append(train_graph)
            test_gs.append(test_graph)
        return train_gs, test_gs

    def positive_sample(self):
        hetero_g = self.process()
        pos_train_gs, pos_test_gs = self.split_train_test(hetero_g=hetero_g)
        return pos_train_gs, pos_test_gs

    def negative_sample(self, hetero_g, k, kFold, pre_edge='l_s'):
        assert k > 0, 'k must beyond 0'
        neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(k, exclude_self_loops=True)
        if pre_edge == 'l_s':
            eid_dict = {etype: hetero_g.edges(etype=etype, form='eid') for etype in
                        ['l_s']}
            neg_eid_sample_dict = neg_sampler(hetero_g, eid_dict)
            neg_eid_sample_dict[('stress', 's_l', 'lnc')] = (neg_eid_sample_dict[('lnc', 'l_s', 'stress')][1],
                                                             neg_eid_sample_dict[('lnc', 'l_s', 'stress')][0])
            neg_eid_sample_dict[('mi', 'm_s', 'stress')] = hetero_g.edge_type_subgraph(['m_s']).find_edges(
                hetero_g.edges(etype='m_s', form='eid'))
            neg_eid_sample_dict[('stress', 's_m', 'mi')] = hetero_g.edge_type_subgraph(['s_m']).find_edges(
                hetero_g.edges(etype='s_m', form='eid'))
        else:
            neg_eid_sample_dict = {
                ('lnc', 'l_s', 'stress'): hetero_g.edge_type_subgraph(['l_s']).find_edges(
                    hetero_g.edges(etype='l_s', form='eid')),
                ('stress', 's_l', 'lnc'): hetero_g.edge_type_subgraph(['s_l']).find_edges(
                    hetero_g.edges(etype='s_l', form='eid'))}
            eid_dict = {etype: hetero_g.edges(etype=etype, form='eid') for etype in
                        ['m_s']}
            neg_eid_sample_dict.update(neg_sampler(hetero_g, eid_dict))
            neg_eid_sample_dict[('stress', 's_m', 'mi')] = (neg_eid_sample_dict[('mi', 'm_s', 'stress')][1],
                                                            neg_eid_sample_dict[('mi', 'm_s', 'stress')][0])

        neg_eid_sample_dict[('lnc', 'l_m', 'mi')] = hetero_g.edge_type_subgraph(['l_m']).find_edges(
            hetero_g.edges(etype='l_m', form='eid'))
        neg_eid_sample_dict[('mi', 'm_l', 'lnc')] = hetero_g.edge_type_subgraph(['m_l']).find_edges(
            hetero_g.edges(etype='m_l', form='eid'))
        neg_eid_sample_dict[('lnc', 'l_l', 'lnc')] = hetero_g.edge_type_subgraph(['l_l']).find_edges(
            hetero_g.edges(etype='l_l', form='eid'))
        neg_eid_sample_dict[('mi', 'm_m', 'mi')] = hetero_g.edge_type_subgraph(['m_m']).find_edges(
            hetero_g.edges(etype='m_m', form='eid'))
        neg_eid_sample_dict['stress', 's_s', 'stress'] = hetero_g.edge_type_subgraph(['s_s']).find_edges(
            hetero_g.edges(etype='s_s', form='eid'))
        neg_hetero_g = dgl.heterograph(neg_eid_sample_dict)
        neg_hetero_g.ndata['h'] = hetero_g.ndata['h']
        neg_train_gs, neg_test_gs = self.split_train_test(hetero_g=neg_hetero_g, kFold=kFold, pre_edge=pre_edge)
        return neg_train_gs, neg_test_gs

    def get_adj(self, g):
        hetero_g = g
        edge_types = self.edge_types
        node_types = self.node_types
        meta_adj = []
        for edge_type in edge_types:
            meta_adj.append(np.array(hetero_g.adj(etype=edge_type).to_dense()))
        adj1 = np.concatenate(
            [meta_adj[0], meta_adj[1], meta_adj[2]], axis=1
        )
        adj2 = np.concatenate(
            [meta_adj[3], meta_adj[4], meta_adj[5]], axis=1
        )
        adj3 = np.concatenate(
            [meta_adj[6], meta_adj[7], meta_adj[8]], axis=1
        )
        adj = np.concatenate(
            [adj1, adj2, adj3], axis=0
        )
        return adj

    @staticmethod
    def set_knn_k(k):
        return k
    def custom_knn_graph_blas(self, x, k, algorithm='bruteforce-blas', dist='cosine'):
        """
        自定义的 _knn_graph_blas 实现，支持最近邻和最不近邻居。
        """
        # 计算距离矩阵
        dist_matrix = torch.cdist(x, x)  # 使用 PyTorch 的 cdist 计算欧几里得距离
        # 找到最近邻
        nearest_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1).indices
        # 找到最不近邻居
        farthest_indices = torch.topk(dist_matrix, k=k, largest=True, dim=1).indices
        # 合并最近邻和最不近邻居
        combined_indices = torch.cat([nearest_indices, farthest_indices], dim=1)
        # 构建图的边
        src = torch.arange(x.shape[0]).unsqueeze(1).repeat(1, 2 * k).view(-1)
        dst = combined_indices.view(-1)
        return src, dst
    def add_noise_knn_graph(self, x, k, algorithm='bruteforce-blas', dist='cosine'):
        """
        自定义的 knn_graph 函数，支持最近邻和最不近邻居。
        """
        # 调用自定义的 _knn_graph_blas
        src, dst = self.custom_knn_graph_blas(x, k, algorithm=algorithm, dist=dist)
        # 构建 DGL 图
        g = dgl.graph((src, dst))
        return g
    def process(self):
        """
        1.异构网络邻接矩阵表示：
        [[S_lnc_lnc, A_lnc_stress, A_lnc_mi],
         [A_lnc_stress.T, S_mi_mi, A_mi_stress.T],
         [A_lnc_mi.T, A_mi_stress, S_mi_mi]]
        :return:
        """
        lnc2stress_link = pd.read_csv('./dataset/adj/lnc2stress_map.csv').values
        mi2lnc_link = pd.read_csv('./dataset/adj/mi2lnc_map.csv').values
        mi2stress_link = pd.read_csv('./dataset/adj/mi2stress_map.csv').values
        lnc2stress_link = self.remove_duplicate_rows(lnc2stress_link)
        mi2lnc_link = self.remove_duplicate_rows(mi2lnc_link)
        mi2stress_link = self.remove_duplicate_rows(mi2stress_link)
        lnc_feats = np.load('./dataset/sim/lnc/lnc_gcrna_arr.npz', allow_pickle=True)['arr_0']
        mi_feats = np.load('./dataset/sim/mi/mi_gcrna_arr.npz', allow_pickle=True)['arr_0']
        stress_feats = np.load('./dataset/sim/stress/stress_vec_array.npz', allow_pickle=True)['arr_0']

        # scalar = StandardScaler()
        # lnc_feats = scalar.fit_transform(lnc_feats)
        # mi_feats = scalar.fit_transform(mi_feats)
        # stress_feats = scalar.fit_transform(stress_feats)

        lnc_feats = torch.tensor(lnc_feats)
        mi_feats = torch.tensor(mi_feats)
        stress_feats = torch.tensor(stress_feats)

        node_feat_dict = {
            'lnc': lnc_feats,
            'mi': mi_feats,
            'stress': stress_feats
        }
        args = set_params.get_args()
        k_of_knn = args.kNN
        # knn_lnc_g = dgl.knn_graph(lnc_feats, k=k_of_knn, dist='cosine')
        # ll_src, ll_dst = knn_lnc_g.edges()
        # 加入噪声
        noise_knn_lnc_g = self.add_noise_knn_graph(lnc_feats, k=k_of_knn)
        ll_src, ll_dst = noise_knn_lnc_g.edges()

        # knn_mi_g = dgl.knn_graph(mi_feats, k=k_of_knn, dist='cosine')
        # mm_src, mm_dst = knn_mi_g.edges()
        # 加入噪声
        noise_knn_mi_g = self.add_noise_knn_graph(mi_feats, k=k_of_knn)
        mm_src, mm_dst = noise_knn_mi_g.edges()

        # knn_stress_g = dgl.knn_graph(stress_feats, k=k_of_knn, dist='cosine')
        # ss_src, ss_dst = knn_stress_g.edges()
        # 加入噪声

        noise_knn_stress_g = self.add_noise_knn_graph(stress_feats, k=k_of_knn, dist='cosine')
        ss_src, ss_dst = noise_knn_stress_g.edges()

        hetero_g = dgl.heterograph({
            ('lnc', 'l_s', 'stress'): (
                torch.tensor(lnc2stress_link[:, 1]), torch.tensor(lnc2stress_link[:, 0])
            ),
            ('stress', 's_l', 'lnc'): (
                torch.tensor(lnc2stress_link[:, 0]), torch.tensor(lnc2stress_link[:, 1])
            ),
            ('mi', 'm_s', 'stress'): (
                torch.tensor(mi2stress_link[:, 1]), torch.tensor(mi2stress_link[:, 0])
            ),
            ('stress', 's_m', 'mi'): (
                torch.tensor(mi2stress_link[:, 0]), torch.tensor(mi2stress_link[:, 1])
            ),
            ('lnc', 'l_m', 'mi'): (
                torch.tensor(mi2lnc_link[:, 1]), torch.tensor(mi2lnc_link[:, 0])
            ),
            ('mi', 'm_l', 'lnc'): (
                torch.tensor(mi2lnc_link[:, 0]), torch.tensor(mi2lnc_link[:, 1])
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
        hetero_g.ndata['h'] = node_feat_dict

        return hetero_g

    def __getitem__(self, idx):
        hetero_g = MYdataset.process(self)
        return hetero_g

    def __len__(self):
        return 1


if __name__ == '__main__':
    data = MYdataset()
    hg = data[0]
    # tr_gs, te_gs = data.split_train_test(hg)
    # tr_g = tr_gs[0]
    # etypes = tr_g.etypes
    # tr_meta_gs = [tr_g.edge_type_subgraph([etype]) for etype in etypes]
    # new_tr_meta_gs = []
    # tr_node_neighbor_table = {}
    # for etype, meta_g in zip(etypes, tr_meta_gs):
    #     new_tr_meta_gs.append(data.get_community_labels(g=meta_g, edge_type=etype))
    #     tr_node_neighbor_table[etype] = data.get_node_neighbors(meta_g, edge_type=etype)
