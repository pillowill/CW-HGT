"""Dataset builder for the two-type miRNA-stress graph (``--DataClass mi_stress_graphs``).
"""

import dgl
import numpy as np
import torch
import pandas as pd
from dgl.data import DGLDataset
from sknetwork.clustering import Louvain
# from sknetwork.utils import membership_matrix
import os, sys

# 获取当前文件的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到 Python 搜索路径
sys.path.append(parent_dir)
from cw_hgt import config


class MYdataset(DGLDataset):
    def __init__(self):
        super(MYdataset, self).__init__(name='mydata')
        self.hg = self.process()
        self.edge_types = self.hg.etypes
        self.node_types = self.hg.ntypes
        self.args = config.get_args()
        self.link_types = [
            ['mi', 'm_s', 'stress'],
            ['stress', 's_m', 'mi'],
            ['mi', 'm_m', 'mi'],
            ['stress', 's_s', 'stress']
        ]

    def split_train_test(self, hetero_g, kFold=5, pre_edge='l_s'):
        np.random.seed(self.args.random_seed)
        # Define the number of edges for each edge type
        num_mi2stress = hetero_g.num_edges('m_s')
        num_m_m = hetero_g.num_edges('m_m')
        num_s_s = hetero_g.num_edges('s_s')
        train_gs = []
        test_gs = []
        # Create folds for each edge type
        num_folds = kFold
        mi2stress = np.arange(num_mi2stress)
        np.random.shuffle(mi2stress)
        mi2stress_folds = np.array_split(np.arange(num_mi2stress), num_folds)

        for fold_idx in range(num_folds):
            # Generate masks for training and testing edges based on the current fold
            train_idx_mi2stress = np.concatenate([mi2stress_folds[i] for i in range(num_folds) if i != fold_idx])

            train_mask_mi2stress = torch.zeros(num_mi2stress, dtype=torch.bool)
            train_mask_mi2stress[train_idx_mi2stress] = True

            test_mask_mi2stress = ~train_mask_mi2stress

            # Add these masks as edge features to the graph
            hetero_g.edges[('mi', 'm_s', 'stress')].data['train_mask_fold_' + str(fold_idx)] = train_mask_mi2stress
            hetero_g.edges[('mi', 'm_s', 'stress')].data['test_mask_fold_' + str(fold_idx)] = test_mask_mi2stress

            # Create training and testing subgraphs for the current fold
            train_edge_dict = {
                ('mi', 'm_s', 'stress'): train_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
                ('stress', 's_m', 'mi'): train_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
                ('mi', 'm_m', 'mi'): np.arange(num_m_m),
                ('stress', 's_s', 'stress'): np.arange(num_s_s),
            }

            test_edge_dict = {
                ('mi', 'm_s', 'stress'): test_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
                ('stress', 's_m', 'mi'): test_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
                ('mi', 'm_m', 'mi'): np.arange(num_m_m),
                ('stress', 's_s', 'stress'): np.arange(num_s_s),
            }

            train_graph = dgl.edge_subgraph(hetero_g, train_edge_dict, relabel_nodes=False)
            test_graph = dgl.edge_subgraph(hetero_g, test_edge_dict, relabel_nodes=False)
            train_gs.append(train_graph)
            test_gs.append(test_graph)
        return train_gs, test_gs

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

    def positive_sample(self):
        hetero_g = self.process()
        pos_train_gs, pos_test_gs = self.split_train_test(hetero_g=hetero_g)
        return pos_train_gs, pos_test_gs

    def negative_sample(self, hetero_g, k, kFold, pre_edge='m_s'):
        assert k > 0, 'k must beyond 0'
        neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(k, exclude_self_loops=True)
        eid_dict = {etype: hetero_g.edges(etype=etype, form='eid') for etype in
                    ['m_s']}
        neg_eid_sample_dict = neg_sampler(hetero_g, eid_dict)
        neg_eid_sample_dict[('stress', 's_m', 'mi')] = (neg_eid_sample_dict[('mi', 'm_s', 'stress')][1],
                                                        neg_eid_sample_dict[('mi', 'm_s', 'stress')][0])
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
            [meta_adj[0], meta_adj[1]], axis=1
        )
        adj2 = np.concatenate(
            [meta_adj[2], meta_adj[3]], axis=1
        )
        adj = np.concatenate(
            [adj1, adj2], axis=0
        )
        return adj

    def process(self):
        """
        1.异构网络邻接矩阵表示：
        [[S_mi_mi, A_mi_stress],
         [A_mi_stress.T, S_stress_stress]]
        :return:
        """
        mi2stress_link = pd.read_csv('./dataset/adj/mi2stress_map.csv').values
        mi2stress_link = self.remove_duplicate_rows(mi2stress_link)
        mi_feats = np.load('./dataset/sim/mi/mi_gcrna_arr.npz', allow_pickle=True)['arr_0']
        stress_feats = np.load('./dataset/sim/stress/stress_vec_array.npz', allow_pickle=True)['arr_0']

        mi_feats = torch.tensor(mi_feats)
        stress_feats = torch.tensor(stress_feats)
        node_feat_dict = {
            'mi': mi_feats,
            'stress': stress_feats
        }
        args = config.get_args()
        k_of_knn = args.kNN
        knn_mi_g = dgl.knn_graph(mi_feats, k=k_of_knn, dist='cosine')
        mm_src, mm_dst = knn_mi_g.edges()

        knn_stress_g = dgl.knn_graph(stress_feats, k=k_of_knn, dist='cosine')
        ss_src, ss_dst = knn_stress_g.edges()

        hetero_g = dgl.heterograph({
            ('mi', 'm_s', 'stress'): (
                torch.tensor(mi2stress_link[:, 1]), torch.tensor(mi2stress_link[:, 0])
            ),
            ('stress', 's_m', 'mi'): (
                torch.tensor(mi2stress_link[:, 0]), torch.tensor(mi2stress_link[:, 1])
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
