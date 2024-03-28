import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import json
from collections import defaultdict
import numpy  as np
import pandas as pd
import math
import copy
import networkx as nx
import pickle
from torch.autograd import Variable
from data_loader import *
train_data_json = '../data/junyi/train_set.json'
test_data_json = '../data/junyi/test_set.json'

class our_adaptive(nn.Module):
    def __init__(self, args, exer_n, student_n, knowledge_n, dim, num_layer, epsilon):
        self.layer_depth = num_layer
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable
        self.epsilon = epsilon
        super(our_adaptive, self).__init__()

        # prediction sub-net
        self.student_emb_bias = nn.Embedding(self.student_n, 1)
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Embedding(self.knowledge_n, self.emb_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.W_1 = nn.Linear(knowledge_n,knowledge_n)
        self.W_0 = nn.Linear(knowledge_n, knowledge_n)


        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        self.graph_init(args)
        self.update = False


    def graph_init(self, args):
        with open(train_data_json, encoding='utf8') as i_f:
            data = json.load(i_f)
        train_data_user_score1, train_data_user_score0 = defaultdict(set), defaultdict(set)
        train_data_item_score1, train_data_item_score0 = defaultdict(set), defaultdict(set)
        for idx, log in enumerate(data):
            u_id = log['user_id']
            i_id = log['exer_id']
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
        sparse_u_i_1 = readTrainSparseMatrix(args, train_data_user_score1, u_d_1, i_d_1, True)
        sparse_i_u_1 = readTrainSparseMatrix(args, train_data_item_score1, u_d_1, i_d_1, False)
        sparse_u_i_0 = readTrainSparseMatrix(args, train_data_user_score0, u_d_0, i_d_0, True)
        sparse_i_u_0 = readTrainSparseMatrix(args, train_data_item_score0, u_d_0, i_d_0, False)

        for i in range(len(u_d_1)):
            u_d_1[i] = [u_d_1[i]]
        for i in range(len(i_d_1)):
            i_d_1[i] = [i_d_1[i]]

        for i in range(len(u_d_0)):
            u_d_0[i] = [u_d_0[i]]
        for i in range(len(i_d_0)):
            i_d_0[i] = [i_d_0[i]]

        self.d_i_train_1 = torch.cuda.FloatTensor(u_d_1)
        self.d_j_train_1 = torch.cuda.FloatTensor(i_d_1)
        self.d_i_train_0 = torch.cuda.FloatTensor(u_d_0)
        self.d_j_train_0 = torch.cuda.FloatTensor(i_d_0)

        self.d_i_train_1 = self.d_i_train_1.expand(-1, self.knowledge_n)
        self.d_j_train_1 = self.d_j_train_1.expand(-1, self.knowledge_n)

        self.d_i_train_0 = self.d_i_train_0.expand(-1, self.knowledge_n)
        self.d_j_train_0 = self.d_j_train_0.expand(-1, self.knowledge_n)

        self.user_item_matrix_1 = sparse_u_i_1.cuda()
        self.item_user_matrix_1 = sparse_i_u_1.cuda()
        self.user_item_matrix_0 = sparse_u_i_0.cuda()
        self.item_user_matrix_0 = sparse_i_u_0.cuda()


    def graph_update(self, args, predicted_results):
        '''
        This part is to update each links
        :return:
        '''
        train_data_user_score_postrue  = defaultdict(set)
        train_data_user_score_negtrue  = defaultdict(set)
        train_data_item_score_postrue  = defaultdict(set)
        train_data_item_score_negtrue  = defaultdict(set)

        pos_interactions = predicted_results[np.where(predicted_results[:, 2] == 1)]
        neg_interactions = predicted_results[np.where(predicted_results[:, 2] == 0)]

        # I only drop a little part of un-trusted interactions!
        neg_interactions_true = neg_interactions[np.where(neg_interactions[:, 3] <= 1-self.epsilon)]
        # I only drop a little part of un-trusted interactions!
        pos_interactions_true = pos_interactions[np.where(pos_interactions[:, 3] >= self.epsilon)]

        for value in pos_interactions_true:
            user = int(value[0])
            item = int(value[1])
            train_data_user_score_postrue[user].add(int(item))
            train_data_item_score_postrue[int(item)].add(user)
        for value in neg_interactions_true:
            user = int(value[0])
            item = int(value[1])
            train_data_user_score_negtrue[user].add(int(item))
            train_data_item_score_negtrue[int(item)].add(user)

        u_d_postrue = readD(args, train_data_user_score_postrue, args.student_n)
        i_d_postrue = readD(args, train_data_item_score_postrue, args.exer_n)
        u_d_negtrue = readD(args, train_data_user_score_negtrue, args.student_n)
        i_d_negtrue = readD(args, train_data_item_score_negtrue, args.exer_n)

        sparse_ui_postrue = readTrainSparseMatrix(args, train_data_user_score_postrue, u_d_postrue, i_d_postrue, True)
        sparse_iu_postrue = readTrainSparseMatrix(args, train_data_item_score_postrue, u_d_postrue, i_d_postrue, False)
        sparse_ui_negtrue = readTrainSparseMatrix(args, train_data_user_score_negtrue, u_d_negtrue, i_d_negtrue, True)
        sparse_iu_negtrue = readTrainSparseMatrix(args, train_data_item_score_negtrue, u_d_negtrue, i_d_negtrue, False)

        self.user_item_matrix_postrue = sparse_ui_postrue.cuda()
        self.user_item_matrix_negtrue = sparse_ui_negtrue.cuda()
        self.item_user_matrix_postrue = sparse_iu_postrue.cuda()
        self.item_user_matrix_negtrue = sparse_iu_negtrue.cuda()

        for i in range(len(u_d_postrue)):
            u_d_postrue[i] = [u_d_postrue[i]]
        for i in range(len(i_d_postrue)):
            i_d_postrue[i] = [i_d_postrue[i]]

        for i in range(len(u_d_negtrue)):
            u_d_negtrue[i] = [u_d_negtrue[i]]
        for i in range(len(i_d_negtrue)):
            i_d_negtrue[i] = [i_d_negtrue[i]]

        self.d_i_train_11 = torch.cuda.FloatTensor(u_d_postrue)
        self.d_j_train_11 = torch.cuda.FloatTensor(i_d_postrue)
        self.d_i_train_01 = torch.cuda.FloatTensor(u_d_negtrue)
        self.d_j_train_01 = torch.cuda.FloatTensor(i_d_negtrue)
        self.update = True

    def graph_representations(self):
        stu_emb = self.student_emb.weight
        exer_emb = self.exercise_emb.weight
        stu_emb_bias = self.student_emb_bias.weight
        stu_emb_bias = stu_emb_bias.reshape(-1,1)
        knowledge_base_emb = self.knowledge_emb.weight

        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = knowledge_base_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        stat_emb = (stu_emb * knowledge_emb).sum(dim=-1, keepdim=False)

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = knowledge_base_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        k_difficulty = (exer_emb * knowledge_emb).sum(dim=-1, keepdim=False)

        if not self.update:
            for _ in range(self.layer_depth):
                gcn1_users_embedding_1 = torch.sparse.mm(self.user_item_matrix_1, k_difficulty) + stat_emb.mul(self.d_i_train_1)
                gcn1_users_embedding_1 = self.W_1(gcn1_users_embedding_1)
                gcn1_items_embedding_1 = torch.sparse.mm(self.item_user_matrix_1, stat_emb) + k_difficulty.mul(self.d_j_train_1)
                gcn1_items_embedding_1 = self.W_1(gcn1_items_embedding_1)
                gcn1_users_embedding_0 = torch.sparse.mm(self.user_item_matrix_0, k_difficulty) + stat_emb.mul(self.d_i_train_0)
                gcn1_users_embedding_0 = self.W_0(gcn1_users_embedding_0)
                gcn1_items_embedding_0 = torch.sparse.mm(self.item_user_matrix_0, stat_emb) + k_difficulty.mul(self.d_j_train_0)
                gcn1_items_embedding_0 = self.W_0(gcn1_items_embedding_0)
                stat_emb = gcn1_users_embedding_1 + gcn1_users_embedding_0
                k_difficulty = gcn1_items_embedding_1 + gcn1_items_embedding_0
        else:
            for _ in range(self.layer_depth):
                gcn1_users_embedding_postrue = torch.sparse.mm(self.user_item_matrix_postrue, k_difficulty) + stat_emb.mul(self.d_i_train_11)
                gcn1_users_embedding_postrue = self.W_1(gcn1_users_embedding_postrue)
                gcn1_items_embedding_postrue = torch.sparse.mm(self.item_user_matrix_postrue, stat_emb) + k_difficulty.mul(self.d_j_train_11)
                gcn1_items_embedding_postrue = self.W_1(gcn1_items_embedding_postrue)
                gcn1_users_embedding_negtrue = torch.sparse.mm(self.user_item_matrix_negtrue, k_difficulty) + stat_emb.mul(self.d_i_train_01)
                gcn1_users_embedding_negtrue = self.W_0(gcn1_users_embedding_negtrue)
                gcn1_items_embedding_negtrue = torch.sparse.mm(self.item_user_matrix_negtrue, stat_emb) + k_difficulty.mul(self.d_j_train_01)
                gcn1_items_embedding_negtrue = self.W_0(gcn1_items_embedding_negtrue)
                stat_emb = gcn1_users_embedding_postrue + gcn1_users_embedding_negtrue
                k_difficulty = gcn1_items_embedding_postrue + gcn1_items_embedding_negtrue
        return stat_emb, k_difficulty, stu_emb_bias

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        stat_emb, k_difficulty, stu_emb_bias = self.graph_representations()
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        stat_emb = torch.sigmoid(stat_emb[stu_id]+stu_emb_bias[stu_id])
        k_difficulty = torch.sigmoid(k_difficulty[input_exercise])
        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        return output_1.view(-1)

    def predict_proficiency_on_concepts(self):
        stat_emb, k_difficulty, stu_emb_bias = self.graph_representations()
        stat_emb = torch.sigmoid(stat_emb+stu_emb_bias)
        return stat_emb

class KaNCD(nn.Module):

    def __init__(self, args, exer_n, student_n, knowledge_n, mf_type, dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(KaNCD, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)


    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb.weight
        exer_emb = self.exercise_emb.weight
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        stat_emb = stat_emb[stu_id]
        k_difficulty = k_difficulty[input_exercise]

        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def predict_proficiency_on_concepts(self):
        stu_emb = self.student_emb.weight
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        return stat_emb

    def predict_knowledge_embeddings(self):
        return self.knowledge_emb

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


def irt2pl(theta, a, b, F=np):
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))

def irf(theta, a, b, c, D=1.702, *, F=np):
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))

irt3pl = irf
