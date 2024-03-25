import json
import torch
import math
import random
import pickle
import time
import pdb
import copy
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from collections import defaultdict
from torch.autograd import Variable


train_data_json = '../data/ASSIST/train_set.json'
test_data_json = '../data/ASSIST/test_set.json'


def obtain_adjency_matrix(args):
    with open(train_data_json, encoding='utf8') as i_f:
        data = json.load(i_f)
    train_data_user_score1,train_data_user_score0 = defaultdict(set), defaultdict(set)
    train_data_item_score1,train_data_item_score0 = defaultdict(set), defaultdict(set)
    for idx, log in enumerate(data):
        u_id = log['user_id'] - 1
        i_id = log['exer_id'] - 1
        if log['score'] == 1:
            train_data_user_score1[u_id].add(int(i_id))
            train_data_item_score1[int(i_id)].add(u_id)
        elif log['score'] == 0:
            train_data_user_score0[u_id].add(int(i_id))
            train_data_item_score0[int(i_id)].add(u_id)
        else:
            assert False, 'rating must be 1 or 0.'

    u_d_1 = readD(args, train_data_user_score1, args.student_n)
    i_d_1 = readD(args, train_data_item_score1, args.exer_n)
    u_d_0 = readD(args, train_data_user_score0, args.student_n)
    i_d_0 = readD(args, train_data_item_score0, args.exer_n)
    sparse_u_i_1 = readTrainSparseMatrix(args, train_data_user_score1,u_d_1, i_d_1,  True)
    sparse_i_u_1 = readTrainSparseMatrix(args, train_data_item_score1,u_d_1, i_d_1, False)
    sparse_u_i_0 = readTrainSparseMatrix(args, train_data_user_score0,u_d_0, i_d_0, True)
    sparse_i_u_0 = readTrainSparseMatrix(args, train_data_item_score0,u_d_0, i_d_0, False)


    return [u_d_1,i_d_1,sparse_u_i_1,sparse_i_u_1], [u_d_0, i_d_0, sparse_u_i_0, sparse_i_u_0]


def readD(args, set_matrix,num_):
    user_d=[]
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)
        user_d.append(len_set)
    return user_d


def readTrainSparseMatrix(args, set_matrix,u_d,i_d, is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    exer_num = args.exer_n
    student_n = args.student_n
    if is_user:
        d_i=u_d
        d_j=i_d
        user_items_matrix_i.append([student_n-1, exer_num-1])
        user_items_matrix_v.append(0)
    else:
        d_i=i_d
        d_j=u_d
        user_items_matrix_i.append([exer_num - 1, student_n - 1])
        user_items_matrix_v.append(0)
    for i in set_matrix:
        len_set=len(set_matrix[i])
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            user_items_matrix_v.append(d_i_j)
    user_items_matrix_i=torch.LongTensor(user_items_matrix_i).cuda()
    user_items_matrix_v=torch.FloatTensor(user_items_matrix_v).cuda()
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)


class EduData(data.Dataset):
    def __init__(self, type='train'):
        super(EduData, self).__init__()
        if type == 'train':
            self.data_file = train_data_json
            self.type = 'train'
        elif type == 'predict':
            self.data_file = test_data_json
            self.type = 'predict'
        else:
            assert False, 'type can only be selected from train or predict'
        with open(self.data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        self.config_file = 'config.txt'
        with open(self.config_file) as i_f:
            i_f.readline()
            student_n, exercise_n, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.student_dim = int(student_n)
        self.exercise_dim = int(exercise_n)

    def load_data(self):
        '''
        if first load, use comment part.
        :return:
        '''
        self.dataset = []
        self.k_ids = []
        for idx, log in enumerate(self.data):
            u_id = log['user_id'] - 1
            e_id = log['exer_id'] - 1
            y = log['score']
            self.dataset.append([u_id, e_id, y])
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            self.k_ids.append(knowledge_emb)

        self.k_ids = torch.cuda.LongTensor(self.k_ids)
        self.data_len = len(self.dataset)

    def __len__(self):
        return  self.data_len

    def __getitem__(self, idx):
        u_id = self.dataset[idx][0]
        i_id = self.dataset[idx][1]
        label = self.dataset[idx][2]
        k_id = self.k_ids[idx]
        return u_id, i_id, k_id, label
