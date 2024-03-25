import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import scipy.linalg as slin
import scipy.sparse as sp
import networkx as nx
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
import glob
import re
import pickle
import math
from tqdm import tqdm
from torch.optim.adam import Adam
import pdb

# Education
with open('config.txt') as i_f:
    i_f.readline()
    student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=exer_n,
                          help='The number for exercise.')
        self.add_argument('--knowledge_n', type=int, default=knowledge_n,
                          help='The number for knowledge concept.')
        self.add_argument('--student_n', type=int, default=student_n,
                          help='The number for student.')
        self.add_argument('--gpu', type=int, default=0,
                          help='The id of gpu, e.g. 0.')
        self.add_argument('--epoch_n', type=int, default=4,
                          help='The epoch number of training')
        self.add_argument('--lr', type=float, default=0.005,
                          help='Learning rate')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the testing set in the training process.')
        self.add_argument('--x_dims', type=int, default=1,  # changed here
                            help='The number of input dimensions: default 1.')
        self.add_argument('--z_dims', type=int, default=1,
                            help='The number of latent variable dimensions: default the same as variable size.')
        self.add_argument('--graph_threshold', type=float, default=0.3,  # 0.3 is good, 0.2 is error prune
                            help='threshold for learned adjacency matrix binarization')
        self.add_argument('--tau_A', type=float, default=0.0,
                            help='coefficient for L-1 norm of A.')
        self.add_argument('--lambda_A', type=float, default=0.,
                            help='coefficient for DAG constraint h(A).')
        self.add_argument('--c_A', type=float, default=1,
                            help='coefficient for absolute value h(A).')
        self.add_argument('--encoder-dropout', type=float, default=0.0,
                            help='Dropout rate (1 - keep probability).')
        self.add_argument('--decoder-dropout', type=float, default=0.0,
                            help='Dropout rate (1 - keep probability).')
        self.add_argument('--encoder-hidden', type=int, default=64,
                            help='Number of hidden units.')
        self.add_argument('--decoder-hidden', type=int, default=64,
                            help='Number of hidden units.')

# def construct_local_map(args):
#     local_map = {
#         'directed_g': build_graph('direct', args.knowledge_n),
#         'undirected_g': build_graph('undirect', args.knowledge_n),
#         'k_from_e': build_graph('k_from_e', args.knowledge_n + args.exer_n),
#         'e_from_k': build_graph('e_from_k', args.knowledge_n + args.exer_n),
#         'u_from_e': build_graph('u_from_e', args.student_n + args.exer_n),
#         'e_from_u': build_graph('e_from_u', args.student_n + args.exer_n),
#     }
#     return local_map

def doa_report(user, item, know_id, predicted_score, predicted_theta):
    '''

    :param user: 1 int
    :param item: 1 int
    :param know_id: [0,1,..., 1] 123*1
    :param predicted_score: 0.7 float
    :param predicted_theta: user on all concepts [0.9, 0.4, ..., ] 123*1
    :return:
    '''
    df = pd.DataFrame({
        "user_id": user,
        "item_id": item,
        "score": predicted_score,
        "theta": predicted_theta,
        "knowledge": know_id
    })

    knowledges = []
    knowledge_item = []
    knowledge_user = []
    knowledge_truth = []
    knowledge_theta = []
    for user, item, score, theta, knowledge in df[["user_id", "item_id", "score", "theta", "knowledge"]].values:
        # 遍历所有数据，转存数据为id信息
        if isinstance(theta, list):
            for i, (theta_i, knowledge_i) in enumerate(zip(theta, knowledge)):
                if knowledge_i == 1: # 如果这个知识点维度被采用，执行以下结果
                    knowledges.append(i)
                    knowledge_item.append(item)
                    knowledge_user.append(user)
                    knowledge_truth.append(score)
                    knowledge_theta.append(theta_i)
        else:  # pragma: no cover
            for i, knowledge_i in enumerate(knowledge):
                if knowledge_i == 1:
                    knowledges.append(i)
                    knowledge_item.append(item)
                    knowledge_user.append(user)
                    knowledge_truth.append(score)
                    knowledge_theta.append(theta)

    knowledge_df = pd.DataFrame({
        "knowledge": knowledges,
        "user_id": knowledge_user,
        "item_id": knowledge_item,
        "score": knowledge_truth,
        "theta": knowledge_theta
    })
    knowledge_ground_truth = []
    knowledge_prediction = []
    # 按照知识点区分所有样本
    for _, group_df in knowledge_df.groupby("knowledge"):
        _knowledge_ground_truth = []
        _knowledge_prediction = []
        # 每一个知识点对应着有若干个产品
        for _, item_group_df in group_df.groupby("item_id"):
            # 确定知识点，确定产品，可以把训练集和测试集涉及到这个产品的评分全部加入进去
            _knowledge_ground_truth.append(item_group_df["score"].values)
            # 确定知识点，确定产品，可以把训练集和测试集涉及到这个知识点的评分全部加入进去
            _knowledge_prediction.append(item_group_df["theta"].values)
        knowledge_ground_truth.append(_knowledge_ground_truth)
        knowledge_prediction.append(_knowledge_prediction)

    return doa_eval(knowledge_ground_truth, knowledge_prediction)


def doa_eval(y_true, y_pred):
    """
    >>> y_true = [[np.array([1, 0, 1])],[np.array([0, 1, 1])]]
    >>> y_pred = [[np.array([.5, .4, .6])],   [np.array([.2, .3, .5])]]
    >>> doa_eval(y_true, y_pred)['doa'] 1.0
    >>> y_pred = [[np.array([.4, .5, .6])],[np.array([.3, .2, .5])]]
    >>> doa_eval(y_true, y_pred)['doa'] 0.5
    """
    doa = []
    doa_support = 0
    z_support = 0
    niubi = []
    laji = []
    for knowledge_label, knowledge_pred in tqdm(zip(y_true, y_pred),
                                                "doa metrics"):
        _doa = 0
        _z = 0
        length = 0

        for label, pred in zip(knowledge_label, knowledge_pred):
            if sum(label) == len(label) or sum(label) == 0:
                continue
            pos_idx = []
            neg_idx = []
            for i, _label in enumerate(label):
                if _label == 1:
                    pos_idx.append(i)
                else:
                    neg_idx.append(i)
            pos_pred = pred[pos_idx]
            neg_pred = pred[neg_idx]
            invalid = 0
            for _pos_pred in pos_pred:
                _doa += len(neg_pred[neg_pred < _pos_pred])
                invalid += len(neg_pred[neg_pred == _pos_pred])
            _z += (len(pos_pred) * len(neg_pred)) #- invalid
            length = len(label)
        if _z > 0:
            if _doa/_z < 0.2:
                laji.append(length)
            if _doa/ _z > 0.8:
                niubi.append(length)
            doa.append(_doa / _z)
            z_support += _z
            doa_support += 1

    return {
        "doa": np.mean(doa),
        "doa_know_support": doa_support,
        "doa_z_support": z_support,
    }

