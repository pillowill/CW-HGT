# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/13 10:38
@Auth ： He Yu
@File ：train_val_test.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import numpy as np
import torch
import dgl


def split_train_test(hetero_g, val_ratio=0.1):
    # Define the number of edges for each edge type
    num_lnc2stress = hetero_g.num_edges('l_s')
    num_mi2lnc = hetero_g.num_edges('m_l')
    num_mi2stress = hetero_g.num_edges('m_s')
    num_l_l = hetero_g.num_edges('l_l')
    num_m_m = hetero_g.num_edges('m_m')
    num_s_s = hetero_g.num_edges('s_s')

    train_gs = []
    val_gs = []
    test_gs = []

    # Create folds for each edge type
    num_folds = 5
    lnc2stress_folds = np.array_split(np.arange(num_lnc2stress), num_folds)
    mi2lnc_folds = np.array_split(np.arange(num_mi2lnc), num_folds)
    mi2stress_folds = np.array_split(np.arange(num_mi2stress), num_folds)

    for fold_idx in range(num_folds):
        # Split each fold into training, validation, and testing indices
        train_idx_lnc2stress = np.concatenate([lnc2stress_folds[i] for i in range(num_folds) if i != fold_idx])
        train_idx_mi2lnc = np.concatenate([mi2lnc_folds[i] for i in range(num_folds) if i != fold_idx])
        train_idx_mi2stress = np.concatenate([mi2stress_folds[i] for i in range(num_folds) if i != fold_idx])

        val_size_lnc2stress = int(len(train_idx_lnc2stress) * val_ratio)
        val_size_mi2lnc = int(len(train_idx_mi2lnc) * val_ratio)
        val_size_mi2stress = int(len(train_idx_mi2stress) * val_ratio)

        val_idx_lnc2stress = train_idx_lnc2stress[:val_size_lnc2stress]
        train_idx_lnc2stress = train_idx_lnc2stress[val_size_lnc2stress:]

        val_idx_mi2lnc = train_idx_mi2lnc[:val_size_mi2lnc]
        train_idx_mi2lnc = train_idx_mi2lnc[val_size_mi2lnc:]

        val_idx_mi2stress = train_idx_mi2stress[:val_size_mi2stress]
        train_idx_mi2stress = train_idx_mi2stress[val_size_mi2stress:]

        train_mask_lnc2stress = torch.zeros(num_lnc2stress, dtype=torch.bool)
        train_mask_lnc2stress[train_idx_lnc2stress] = True
        val_mask_lnc2stress = torch.zeros(num_lnc2stress, dtype=torch.bool)
        val_mask_lnc2stress[val_idx_lnc2stress] = True

        train_mask_mi2lnc = torch.zeros(num_mi2lnc, dtype=torch.bool)
        train_mask_mi2lnc[train_idx_mi2lnc] = True
        val_mask_mi2lnc = torch.zeros(num_mi2lnc, dtype=torch.bool)
        val_mask_mi2lnc[val_idx_mi2lnc] = True

        train_mask_mi2stress = torch.zeros(num_mi2stress, dtype=torch.bool)
        train_mask_mi2stress[train_idx_mi2stress] = True
        val_mask_mi2stress = torch.zeros(num_mi2stress, dtype=torch.bool)
        val_mask_mi2stress[val_idx_mi2stress] = True

        test_mask_lnc2stress = ~train_mask_lnc2stress & ~val_mask_lnc2stress
        test_mask_mi2lnc = ~train_mask_mi2lnc & ~val_mask_mi2lnc
        test_mask_mi2stress = ~train_mask_mi2stress & ~val_mask_mi2stress

        # Add these masks as edge features to the graph
        hetero_g.edges[('lnc', 'l_s', 'stress')].data[f'train_mask_fold_{fold_idx}'] = train_mask_lnc2stress
        hetero_g.edges[('lnc', 'l_s', 'stress')].data[f'val_mask_fold_{fold_idx}'] = val_mask_lnc2stress
        hetero_g.edges[('lnc', 'l_s', 'stress')].data[f'test_mask_fold_{fold_idx}'] = test_mask_lnc2stress

        hetero_g.edges[('mi', 'm_l', 'lnc')].data[f'train_mask_fold_{fold_idx}'] = train_mask_mi2lnc
        hetero_g.edges[('mi', 'm_l', 'lnc')].data[f'val_mask_fold_{fold_idx}'] = val_mask_mi2lnc
        hetero_g.edges[('mi', 'm_l', 'lnc')].data[f'test_mask_fold_{fold_idx}'] = test_mask_mi2lnc

        hetero_g.edges[('mi', 'm_s', 'stress')].data[f'train_mask_fold_{fold_idx}'] = train_mask_mi2stress
        hetero_g.edges[('mi', 'm_s', 'stress')].data[f'val_mask_fold_{fold_idx}'] = val_mask_mi2stress
        hetero_g.edges[('mi', 'm_s', 'stress')].data[f'test_mask_fold_{fold_idx}'] = test_mask_mi2stress

        # Create training, validation, and testing subgraphs for the current fold
        train_edge_dict = {
            ('lnc', 'l_s', 'stress'): train_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
            ('stress', 's_l', 'lnc'): train_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
            ('mi', 'm_s', 'stress'): train_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
            ('stress', 's_m', 'mi'): train_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
            ('lnc', 'l_m', 'mi'): np.arange(num_mi2lnc),
            ('mi', 'm_l', 'lnc'): np.arange(num_mi2lnc),
            ('lnc', 'l_l', 'lnc'): np.arange(num_l_l),
            ('mi', 'm_m', 'mi'): np.arange(num_m_m),
            ('stress', 's_s', 'stress'): np.arange(num_s_s),
        }

        val_edge_dict = {
            ('lnc', 'l_s', 'stress'): val_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
            ('stress', 's_l', 'lnc'): val_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
            ('mi', 'm_s', 'stress'): val_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
            ('stress', 's_m', 'mi'): val_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
            ('lnc', 'l_m', 'mi'): np.arange(num_mi2lnc),
            ('mi', 'm_l', 'lnc'): np.arange(num_mi2lnc),
            ('lnc', 'l_l', 'lnc'): np.arange(num_l_l),
            ('mi', 'm_m', 'mi'): np.arange(num_m_m),
            ('stress', 's_s', 'stress'): np.arange(num_s_s),
        }

        test_edge_dict = {
            ('lnc', 'l_s', 'stress'): test_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
            ('stress', 's_l', 'lnc'): test_mask_lnc2stress.nonzero(as_tuple=True)[0].tolist(),
            ('mi', 'm_s', 'stress'): test_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
            ('stress', 's_m', 'mi'): test_mask_mi2stress.nonzero(as_tuple=True)[0].tolist(),
            ('lnc', 'l_m', 'mi'): np.arange(num_mi2lnc),
            ('mi', 'm_l', 'lnc'): np.arange(num_mi2lnc),
            ('lnc', 'l_l', 'lnc'): np.arange(num_l_l),
            ('mi', 'm_m', 'mi'): np.arange(num_m_m),
            ('stress', 's_s', 'stress'): np.arange(num_s_s),
        }

        train_graph = dgl.edge_subgraph(hetero_g, train_edge_dict, relabel_nodes=False)
        val_graph = dgl.edge_subgraph(hetero_g, val_edge_dict, relabel_nodes=False)
        test_graph = dgl.edge_subgraph(hetero_g, test_edge_dict, relabel_nodes=False)

        train_gs.append(train_graph)
        val_gs.append(val_graph)
        test_gs.append(test_graph)

    return train_gs, val_gs, test_gs
