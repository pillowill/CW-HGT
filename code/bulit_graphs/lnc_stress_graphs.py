# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/19 15:02
@Auth ： He Yu
@File ：lnc_stress_graphs.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import dgl
import numpy as np
import torch
import pandas as pd
from dgl.data import DGLDataset
from sknetwork.clustering import Louvain
# from sknetwork.utils import membership_matrix


class MYdataset(DGLDataset):
    def __init__(self):
        super(MYdataset, self).__init__(name='mydata')
        self.hg = self.process()
        self.edge_types = self.hg.etypes
        self.node_types = self.hg.ntypes
        self.link_types = [
            ['lnc', 'l_s', 'stress'],
            ['stress', 's_l', 'lnc'],
            ['lnc', 'l_l', 'lnc'],
            ['stress', 's_s', 'stress']
        ]

    @staticmethod
    def split_train_test(hetero_g, kFold=5, pre_edge='l_s'):
        # Define the number of edges for each edge type
        num_lnc2stress = hetero_g.num_edges('l_s')
        num_l_l = hetero_g.num_edges('l_l')
        num_s_s = hetero_g.num_edges('s_s')
        train_gs = []
        test_gs = []
        # Create folds for each edge type
        num_folds = kFold
        lnc2stress_folds = np.array_split(np.arange(num_lnc2stress), num_folds)

        for fold_idx in range(num_folds):
            # Generate masks for training and testing edges based on the current fold
            train_idx_lnc2stress = np.concatenate([lnc2stress_folds[i] for i in range(num_folds) if i != fold_idx])

            train_mask_lnc2stress = torch.zeros(num_lnc2stress, dtype=torch.bool)
            train_mask_lnc2stress[train_idx_lnc2stress] = True

            test_mask_lnc2stress = ~train_mask_lnc2stress

            # Add these masks as edge features to the graph
            hetero_g.edges[('lnc', 'l_s', 'stress')].data['train_mask_fold_' + str(fold_idx)] = train_mask_lnc2stress
            hetero_g.edges[('lnc', 'l_s', 'stress')].data['test_mask_fold_' + str(fold_idx)] = test_mask_lnc2stress

            # Create training and testing subgraphs for the current fold
            train_edge_dict = {
                ('lnc', 'l_s', 'stress'): train_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
                ('stress', 's_l', 'lnc'): train_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
                ('lnc', 'l_l', 'lnc'): np.arange(num_l_l),
                ('stress', 's_s', 'stress'): np.arange(num_s_s),
            }

            test_edge_dict = {
                ('lnc', 'l_s', 'stress'): test_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
                ('stress', 's_l', 'lnc'): test_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
                ('lnc', 'l_l', 'lnc'): np.arange(num_l_l),
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

    def negative_sample(self, hetero_g, k, kFold, pre_edge='l_s'):
        assert k > 0, 'k must beyond 0'
        neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(k, exclude_self_loops=True)
        eid_dict = {etype: hetero_g.edges(etype=etype, form='eid') for etype in
                    ['l_s']}
        neg_eid_sample_dict = neg_sampler(hetero_g, eid_dict)
        neg_eid_sample_dict[('stress', 's_l', 'lnc')] = (neg_eid_sample_dict[('lnc', 'l_s', 'stress')][1],
                                                         neg_eid_sample_dict[('lnc', 'l_s', 'stress')][0])
        neg_eid_sample_dict[('lnc', 'l_l', 'lnc')] = hetero_g.edge_type_subgraph(['l_l']).find_edges(
            hetero_g.edges(etype='l_l', form='eid'))
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
        [[S_lnc_lnc, A_lnc_stress],
         [A_lnc_stress.T, S_stress_stress]]
        :return:
        """
        lnc2stress_link = pd.read_csv('./dataset/adj/lnc2stress_map.csv').values
        lnc2stress_link = self.remove_duplicate_rows(lnc2stress_link)
        lnc_feats = np.load('./dataset/sim/lnc/lnc_gcrna_arr.npz', allow_pickle=True)['arr_0']
        stress_feats = np.load('./dataset/sim/stress/stress_vec_array.npz', allow_pickle=True)['arr_0']

        lnc_feats = torch.tensor(lnc_feats)
        stress_feats = torch.tensor(stress_feats)
        node_feat_dict = {
            'lnc': lnc_feats,
            'stress': stress_feats
        }
        k_of_knn = 11
        knn_lnc_g = dgl.knn_graph(lnc_feats, k=k_of_knn, dist='cosine')
        ll_src, ll_dst = knn_lnc_g.edges()

        knn_stress_g = dgl.knn_graph(stress_feats, k=k_of_knn, dist='cosine')
        ss_src, ss_dst = knn_stress_g.edges()

        hetero_g = dgl.heterograph({
            ('lnc', 'l_s', 'stress'): (
                torch.tensor(lnc2stress_link[:, 1]), torch.tensor(lnc2stress_link[:, 0])
            ),
            ('stress', 's_l', 'lnc'): (
                torch.tensor(lnc2stress_link[:, 0]), torch.tensor(lnc2stress_link[:, 1])
            ),
            ('lnc', 'l_l', 'lnc'): (
                ll_src, ll_dst
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
