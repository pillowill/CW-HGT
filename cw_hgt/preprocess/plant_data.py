"""Offline preprocessing: turns the raw plant lncRNA/miRNA/stress tables into
the ``dataset/adj`` and ``dataset/sim`` artefacts consumed by the graph
builders.

Requires the optional extras ``fasttext``, ``requests`` and the external
``RNA`` package; see docs/data.md.
"""

from collections import defaultdict
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from RNA import RNA
import fasttext.util
import fasttext.FastText as FT
import requests
from cw_hgt.preprocess.cgr import CGR


def convert2ids(df, old_col_n1, old_col_n2, new_col_n1, new_col_n2):
    df_ids = df.copy(deep=True)
    df_ids[new_col_n1] = df_ids[old_col_n1].astype('category').cat.codes
    df_ids[new_col_n2] = df_ids[old_col_n2].astype('category').cat.codes
    result_df = pd.concat(
        [df.loc[:, [old_col_n1, old_col_n2]], df_ids.loc[:, [new_col_n1, new_col_n2]]],
        axis=1
    )

    return result_df


def add_bracket(string):
    # 使用正则表达式匹配左括号但不匹配右括号的字符串
    pattern = r'\([^)]*$'
    if re.search(pattern, string):
        # 如果匹配成功，则在字符串末尾添加右括号
        string += ')'
    return string


def convert_name2id(id_df1, id_df2, link_df):
    tmp_link_df = link_df.copy(deep=True)
    id_dict1 = {}
    id_dict2 = {}
    for i in range(len(id_df1)):
        id_dict1[str(id_df1.iloc[i, 0]).strip()] = id_df1.iloc[i, 1]
    for i in range(len(id_df2)):
        id_dict2[str(id_df2.iloc[i, 0]).strip()] = id_df2.iloc[i, 1]
    for i in range(len(tmp_link_df)):
        name1 = add_bracket(tmp_link_df.iloc[i, 0]).strip()
        name2 = add_bracket(tmp_link_df.iloc[i, 1]).strip()
        id1 = id_dict1[name1]
        id2 = id_dict2[name2]
        tmp_link_df.iloc[i, 0] = id1
        tmp_link_df.iloc[i, 1] = id2
    return tmp_link_df


class Plant_data_process(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.mi_id_path = './dataset/mi_ids.csv'
        self.lnc_id_path = './dataset/lnc_ids.csv'
        self.stress_id_path = './dataset/stress_ids.csv'
        self.mi_info_path = './dataset/mi_info.txt'
        self.lnc_info_path = './dataset/lnc_info.txt'
        self.mi_stress_mapped_path = './dataset/adj/mi2stress_map.csv'
        self.lnc_stress_mapped_path = './dataset/adj/lnc2stress_map.csv'
        self.mi_lnc_mapped_path = './dataset/adj/mi2lnc_map.csv'
        self.ncRNA_types = ['miRNA', 'lncRNA']
        self.link_data = pd.read_excel(data_path, dtype=str)
        self.species = ['Arabidopsis thaliana', 'Gossypium hirsutum', 'Hordeum vulgare', 'Medicago truncatula',
                        'Oryza sativa', 'Populus trichocarpa', 'Triticum aestivum', 'Zea mays']

    def split2_nc_stress(self):
        for nc_type in self.ncRNA_types:
            pncstress_df = self.link_data
            # 步骤1: 取出type_stress为abiotic的行以及type_ncRNA为miRNA的行
            df_abiotic = pncstress_df[pncstress_df['type_stress'] == 'abiotic'].reset_index(drop=True)
            df_abiotic = df_abiotic[df_abiotic['type_ncRNA'] == nc_type].reset_index(drop=True)
            # 步骤2: 过滤掉sequence为空的行
            df_abiotic = df_abiotic[df_abiotic['sequence'].notna()]
            # 步骤3: 按species分组，得到各物种的DataFrame
            grouped_species = df_abiotic.groupby('species')
            # 创建一个字典，键是物种名，值是对应的DataFrame
            species_dfs = {species: group for species, group in grouped_species}
            for k, v in species_dfs.items():
                for s in self.species:
                    s = s.strip()
                    if s == str(v.iloc[0, 5]).strip():
                        save_path = f'./dataset/link_data/2_useful_link_data/{s}/{s}_{nc_type}_stress.csv'
                        if os.path.exists(f'./dataset/link_data/2_useful_link_data/{s}'):
                            v.to_csv(save_path, index=False, encoding='utf-8')
                        else:
                            os.makedirs(f'./dataset/link_data/2_useful_link_data/{s}')
                            v.to_csv(save_path, index=False, encoding='utf-8')

    def split2_nc_fa(self):  # 从每个类型的ncRNA关联数据中分离ncRNA及其序列，写入响应的fa文件中
        for nc_type in self.ncRNA_types:
            for specie in self.species:
                specie_nc_fa = {}
                nc_path = f'./dataset/link_data/2_useful_link_data/{specie}/{specie}_{nc_type}_stress.csv'
                nc_df = pd.read_csv(nc_path, usecols=[3, 9, 12])
                for i in range(len(nc_df)):
                    nc_name = nc_df.iloc[i, 0]
                    nc_seq = nc_df.iloc[i, 2]
                    if str(nc_seq) == 'nan' or '.' in str(nc_seq):
                        nc_seq = nc_df.iloc[i, 1]
                    specie_nc_fa[nc_name] = str(nc_seq).replace(' ', '')
                with open(f'./dataset/link_data/2_useful_link_data/{specie}/{specie}_{nc_type}.fa', 'w') as fw:
                    for k, v in specie_nc_fa.items():
                        fw.write(f'>{k}\n{v}\n')

    def merge_link_data(self):  # 将使用的8个物种的关联数据进行连接
        my_species = self.species
        mi_lnc_link_lst = []
        mi_stress_link_lst = []
        lnc_stress_link_lst = []
        for species in my_species:
            mi_stress_file = f'./dataset/link_data/2_useful_link_data/{species}/{species}_miRNA_stress.csv'
            lnc_stress_file = f'./dataset/link_data/2_useful_link_data/{species}/{species}_lncRNA_stress.csv'
            mi_lnc_file = f'./dataset/link_data/2_useful_link_data/{species}/mi_lnc.txt'
            mi_stress_link = pd.read_csv(mi_stress_file, usecols=[1, 3])
            lnc_stress_link = pd.read_csv(lnc_stress_file, usecols=[1, 3])
            mi_lnc_link = pd.read_csv(mi_lnc_file, header=1, sep='\t', usecols=[0, 1])
            mi_stress_link_lst.append(mi_stress_link)
            lnc_stress_link_lst.append(lnc_stress_link)
            mi_lnc_link_lst.append(mi_lnc_link)
        mi_stress_links = pd.concat(mi_stress_link_lst, ignore_index=True)
        lnc_stress_links = pd.concat(lnc_stress_link_lst, ignore_index=True)
        mi_lnc_links = pd.concat(mi_lnc_link_lst, ignore_index=True)
        return mi_stress_links, lnc_stress_links, mi_lnc_links

    def id_mapping(self):
        mi_stress_links, lnc_stress_links, mi_lnc_links = self.merge_link_data()
        # 接下来对miRNA、lncRNA以及stress进行编号，并使用编号表示数据
        mi_stress_ids = convert2ids(mi_stress_links, 'stress_treatments', 'nameOf_nc', 'stress_id', 'mi_id')
        lnc_stress_ids = convert2ids(lnc_stress_links, 'stress_treatments', 'nameOf_nc', 'stress_id', 'lnc_id')
        # 对stress进行编号
        stress_df = pd.concat([mi_stress_ids, lnc_stress_ids])
        stress_df['stress_id'] = stress_df['stress_treatments'].astype('category').cat.codes
        # 将编号后的数据进行保存
        mi_ids = mi_stress_ids.iloc[:, [1, 3]].drop_duplicates(subset=['mi_id']).sort_values(by='mi_id')
        lnc_ids = lnc_stress_ids.iloc[:, [1, 3]].drop_duplicates(subset=['lnc_id']).sort_values(by='lnc_id')
        stress_ids = stress_df.loc[:, ['stress_treatments', 'stress_id']].drop_duplicates(
            subset=['stress_id']).sort_values(by='stress_id')
        mi_ids.to_csv(self.mi_id_path, index=False)
        lnc_ids.to_csv(self.lnc_id_path, index=False)
        stress_ids.to_csv(self.stress_id_path, index=False)
        # 使用编号对链接数据进行修改
        mi_stress_mapped = convert_name2id(stress_ids, mi_ids, mi_stress_links)
        lnc_stress_mapped = convert_name2id(stress_ids, lnc_ids, lnc_stress_links)
        mi_lnc_mapped = convert_name2id(mi_ids, lnc_ids, mi_lnc_links)
        mi_stress_mapped.to_csv(self.mi_stress_mapped_path, index=False)
        lnc_stress_mapped.to_csv(self.lnc_stress_mapped_path, index=False)
        mi_lnc_mapped.to_csv(self.mi_lnc_mapped_path, index=False)

    def split_nc_info(self):  # 提取所需使用的ncRNA的信息，即名称、编号、序列、二级结构
        pncstress_df = self.link_data
        pncstress_df = pncstress_df[pncstress_df['sequence'].notna()]
        ncRNA_id_files = [self.mi_id_path, self.lnc_id_path]
        ncRNA_info_files = [self.mi_info_path, self.lnc_info_path]
        for ncRNA_id_file, ncRNA_info_file in zip(ncRNA_id_files, ncRNA_info_files):
            ncRNA_names = pd.read_csv(ncRNA_id_file)
            write_nc = []
            with open(ncRNA_info_file, 'w') as fw:
                for k in tqdm(range(len(ncRNA_names)), desc='getting ncRNA info'):
                    nc_name = ncRNA_names.iloc[k, 0]
                    nc_id = ncRNA_names.iloc[k, 1]
                    for i in range(len(pncstress_df)):
                        cur_nc_name = pncstress_df.iloc[i, 3]
                        if cur_nc_name == nc_name:
                            if cur_nc_name in write_nc:
                                continue
                            if 'mi' in ncRNA_id_file:
                                cur_nc_seq = str(pncstress_df.iloc[i, 11]).replace(' ', '')
                                cur_nc_fold = pncstress_df.iloc[i, 13]
                                if str(cur_nc_seq) == 'nan':
                                    cur_nc_seq = pncstress_df.iloc[i, 9]
                                    cur_nc_fold = pncstress_df.iloc[i, 10]
                                    if str(cur_nc_fold) == 'nan':
                                        cur_nc_fold, energy = RNA.fold(cur_nc_seq.replace('T', 'U'))
                            else:
                                cur_nc_seq = str(pncstress_df.iloc[i, 12]).replace('</br>', '').replace(' ', '')
                                cur_nc_fold = pncstress_df.iloc[i, 13]
                                if '>' in str(cur_nc_seq):
                                    print(nc_name)
                                    print(cur_nc_seq)
                                    cur_nc_seq = str(cur_nc_seq).split('\n')[1].replace(' ', '')
                                if str(cur_nc_seq) == 'nan':
                                    cur_nc_seq = pncstress_df.iloc[i, 9]
                                    cur_nc_fold = pncstress_df.iloc[i, 10]
                                    if str(cur_nc_fold) == 'nan':
                                        cur_nc_fold, energy = RNA.fold(cur_nc_seq.replace('T', 'U'))
                            write_nc.append(nc_name)
                            fw.write(
                                str(nc_name) + '\t' + str(nc_id) + '\t' + str(cur_nc_seq) + '\t' + str(
                                    cur_nc_fold) + '\n')  # 提取所需使用的

    def cal_RNA_CGR(self, nc_info_path, grid_size=2 ** 3,
                    outer_representation=np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])):
        # 输入ncRNA的信息文件，计算ncRNA的CGR特征，并返回特征矩阵和相似矩阵
        nc_info = pd.read_csv(nc_info_path, sep='\t', header=None)
        num_nc = len(nc_info)
        gcrna_dim = 3 * grid_size * grid_size  # x, y, z
        gcrna_vec_arr = np.zeros((num_nc, gcrna_dim))
        for i in tqdm(range(num_nc), desc='calculating ncRNA GCR'):
            nc_seq = str(nc_info.iloc[i, 2]).upper().replace('T', 'U').replace("N", "")
            nc_fold = nc_info.iloc[i, 3]
            if ('Chr' in str(nc_fold)) or ('nan' == str(nc_fold)) or (str(nc_fold).isnumeric()):
                nc_fold = RNA.fold(nc_seq)[0]
            cgrna_vec = CGR(nc_seq, seq_type="RNA", outer_representation=outer_representation,
                            rna_2structure=nc_fold).grid_representation(grid_size=grid_size).reshape(1, -1)
            gcrna_vec_arr[i, :] = cgrna_vec
        gcrna_sim = np.corrcoef(gcrna_vec_arr)
        return gcrna_vec_arr, gcrna_sim

    @staticmethod
    def cal_Stress_feat():
        ft = FT.load_model('./fastText_PreTrained/cc.en.300.bin')
        ft = fasttext.util.reduce_model(ft, 100)
        stress2vec = {}
        stress_words = pd.read_csv('./dataset/stress_ids.csv')
        stress_vec = np.zeros((len(stress_words), 100))
        for i in range(len(stress_words)):
            word = str(stress_words.iloc[i, 0]).strip()
            vec = ft.get_word_vector(word)
            stress2vec[word] = vec
            stress_vec[i, :] = vec
        np.savez('./dataset/sim/stress/stress_vec_ft.npz', **stress2vec)
        np.savez('./dataset/sim/stress/stress_vec_array.npz', stress_vec)


class Homo_data_process(object):
    def __init__(self):
        self.l_d_arr = np.loadtxt('./dataset/homo/dataset1/lnc_dis_association.txt')
        self.l_m_arr = np.loadtxt('./dataset/homo/dataset1/lnc_mi_interaction.txt')
        self.m_d_arr = np.loadtxt('./dataset/homo/dataset1/mi_dis.txt')
        self.l_names = np.loadtxt('./dataset/homo/dataset1/yuguoxian_lncRNA_name.txt')
        self.m_names = np.loadtxt('./dataset/homo/dataset1/miRNA-names.txt')
        self.d_names = np.loadtxt('./dataset/homo/dataset1/disease_name.txt')
        pass

    @staticmethod
    def get_row_col_link(link_arr):
        link_tuple = np.nonzero(link_arr)
        row_ids = link_tuple[0]
        col_ids = link_tuple[1]
        return row_ids, col_ids

    @staticmethod
    def get_rna_seq(rna_names):
        for rna_n in rna_names:
            url = f'https://rnacentral.org/api/v1/rna/?description={rna_n}'
            response = requests.get(url)


if __name__ == '__main__':
    plant_data_path = './dataset/pncStress/All_PncStress.xls'
    plant_data_tooler = Plant_data_process(data_path=plant_data_path)
    plant_data_tooler.split2_nc_stress()
    plant_data_tooler.split2_nc_fa()
    plant_data_tooler.id_mapping()
    plant_data_tooler.split_nc_info()
    plant_data_tooler.split_nc_info()
    # 计算ncRNA的CGR特征
    outer_representation = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [0, 0]])
    mi_gcrna_arr, mi_gcrna_sim = plant_data_tooler.cal_RNA_CGR(
        nc_info_path=plant_data_tooler.mi_info_path, outer_representation=outer_representation
    )
    lnc_gcrna_arr, lnc_gcrna_sim = plant_data_tooler.cal_RNA_CGR(
        nc_info_path=plant_data_tooler.lnc_info_path, outer_representation=outer_representation
    )
    np.savez('./dataset/sim/mi/mi_gcrna_sim.npz', mi_gcrna_sim)
    np.savez('./dataset/sim/lnc/lnc_gcrna_sim.npz', lnc_gcrna_sim)
    np.savez('./dataset/sim/mi/mi_gcrna_arr.npz', mi_gcrna_arr)
    np.savez('./dataset/sim/lnc/lnc_gcrna_arr.npz', lnc_gcrna_arr)
    # 计算得到非生物胁迫的语义特征
    plant_data_tooler.cal_Stress_feat()

