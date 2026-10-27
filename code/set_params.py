# -*- coding: utf-8 -*-
"""
@Time ： 2024/9/9 16:49
@Auth ： He Yu
@File ：set_params.py
@IDE ：PyCharm
@Function ：Function of the script
"""
import argparse
import sys
from bulit_graphs import lnc_mi_stress_graphs, lnc_stress_graphs, mi_stress_graphs


def get_args():
    # epoch; 100; 200
    # random_seed: l-s: 12312, m-s: 123
    # 数据集 预测类型 游走类型 游走长度 epoch knn 随机种子 layers headers
    init_args = ['lnc_mi_stress_graphs', 'l_s', 'CW', 0, 200, 15, 123, 0, 0]  # 尝试一下epoch300
    parser = argparse.ArgumentParser(description='CW-GNN')
    # parameters of dataset
    parser.add_argument('--DataClass', type=str, default=init_args[0],
                        choices=['lnc_mi_stress_graphs', 'lnc_stress_graphs', 'mi_stress_graphs'])
    parser.add_argument('--random_seed', type=int, default=init_args[6])
    parser.add_argument('--pre_edge_type', type=str, default=init_args[1], choices=['l_s', 'm_s'],
                        help='to choose the prediction edge type in Graph')
    parser.add_argument('--num_lnc', type=int, default=175)
    parser.add_argument('--num_mi', type=int, default=500)
    parser.add_argument('--num_stress', type=int, default=50)
    parser.add_argument('--seed', type=int, default=3047, help='fix random seed if needed')
    parser.add_argument('--num_NegativeSample', type=int, default=1, help='construct n-times negative sample graph')
    parser.add_argument('--kNN', type=int, default=init_args[5])
    parser.add_argument('--kFold', type=int, default=10)
    # parameters of walk algorithm
    parser.add_argument('--n_workers', type=int, default=6,
                        help='number of CPU processes for causal walk algorithm')
    parser.add_argument('--walk_type', type=str, default=init_args[2], choices=['None', 'CW', 'N2VRW', 'RW'])
    parser.add_argument('--p', type=float, default=4)
    parser.add_argument('--q', type=int, default=3)
    parser.add_argument('--NumMultiWalk', type=int, default=1, help='the number of multi-walk')
    parser.add_argument('--WalkLength', type=int, default=init_args[3], help='the length of walk')
    parser.add_argument('--WalkSavePath', type=str,
                        default=f'./trained_models/cache_walks/{init_args[0]}/{init_args[1]}/single_path/{init_args[2]}'
                        , help='the save path of walk, json file')
    # parameters of GNN algorithm
    parser.add_argument('--HidDim', type=int, default=128)
    parser.add_argument('--OutDim', type=int, default=128)
    parser.add_argument('--nLayers', type=int, default=init_args[7])
    parser.add_argument('--nHeaders', type=int, default=init_args[8])
    parser.add_argument('--ModelSavePath', type=str,
                        default=f'./trained_models/{init_args[0]}/{init_args[1]}/{init_args[2]}/single_path/outer_path',
                        # default='./trained_models/lnc_mi_stress_graphs/l_s/CW-GCN/',
                        help='weight of cf loss')
    parser.add_argument('--isHetero', type=bool, default=True, choices=[True, False],
                        help='to help choice hetero GNN or homo GNN')
    # parameters of model training
    parser.add_argument('--epochs', type=int, default=init_args[4])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument('--name', type=str, default='debug', help='name for this run for logging')
    args = parser.parse_args()
    args.argv = sys.argv

    return args


if __name__ == '__main__':
    args = get_args()
