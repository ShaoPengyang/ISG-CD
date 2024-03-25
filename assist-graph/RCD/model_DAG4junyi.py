import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import json
from collections import defaultdict
import numpy as np
from numpy import prod
import math
import copy
import networkx as nx
import pickle
from torch.autograd import Variable
from data_loader import *

class KaNCD_hyperRGCN(nn.Module):
    def __init__(self, args, exer_n, student_n, knowledge_n, mf_type, dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(KaNCD_hyperRGCN, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.student_emb_bias = nn.Embedding(self.student_n, 1)
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

        self.Linear_encoder = nn.Linear(self.knowledge_n, self.knowledge_n)
        # self.mu_linear_encoder = nn.Linear(self.emb_dim, self.emb_dim)
        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

        latent_dim = knowledge_n
        adj_matrix1, adj_matrix0 = obtain_adjency_matrix(args)
        d_i_train_1, d_j_train_1, sparse_u_i_1, sparse_i_u_1 = adj_matrix1[0], adj_matrix1[1], adj_matrix1[2], \
                                                               adj_matrix1[3]
        d_i_train_0, d_j_train_0, sparse_u_i_0, sparse_i_u_0 = adj_matrix0[0], adj_matrix0[1], adj_matrix0[2], \
                                                               adj_matrix0[3]

        for i in range(len(d_i_train_1)):
            d_i_train_1[i] = [d_i_train_1[i]]
        for i in range(len(d_j_train_1)):
            d_j_train_1[i] = [d_j_train_1[i]]

        for i in range(len(d_i_train_0)):
            d_i_train_0[i] = [d_i_train_0[i]]
        for i in range(len(d_j_train_0)):
            d_j_train_0[i] = [d_j_train_0[i]]

        self.d_i_train_1 = torch.cuda.FloatTensor(d_i_train_1)
        self.d_j_train_1 = torch.cuda.FloatTensor(d_j_train_1)
        self.d_i_train_0 = torch.cuda.FloatTensor(d_i_train_0)
        self.d_j_train_0 = torch.cuda.FloatTensor(d_j_train_0)

        self.d_i_train_1 = self.d_i_train_1.expand(-1, latent_dim)
        self.d_j_train_1 = self.d_j_train_1.expand(-1, latent_dim)
        self.user_item_matrix_1 = sparse_u_i_1
        self.item_user_matrix_1 = sparse_i_u_1

        self.d_i_train_0 = self.d_i_train_0.expand(-1, latent_dim)
        self.d_j_train_0 = self.d_j_train_0.expand(-1, latent_dim)
        self.user_item_matrix_0 = sparse_u_i_0
        self.item_user_matrix_0 = sparse_i_u_0

        self.loss_function = nn.BCELoss(reduction='mean')
        self.temp_node = 2.0

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb.weight
        stat_emb_bias = self.student_emb_bias.weight
        exer_emb = self.exercise_emb.weight
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = self.stat_full(stu_emb * knowledge_emb).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':
            k_difficulty = (exer_emb * knowledge_emb).sum(dim=-1, keepdim=False)  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = self.k_diff_full(exer_emb * knowledge_emb).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        gcn1_users_embedding_1 = (torch.sparse.mm(self.user_item_matrix_1, k_difficulty) + stat_emb.mul(
            self.d_i_train_1))  # *2. #+ users_embedding
        gcn1_items_embedding_1 = (torch.sparse.mm(self.item_user_matrix_1, stat_emb) + k_difficulty.mul(
            self.d_j_train_1))  # *2. #+ items_embedding

        gcn2_users_embedding_1 = (
                torch.sparse.mm(self.user_item_matrix_1, gcn1_items_embedding_1) + gcn1_users_embedding_1.mul(
            self.d_i_train_1))  # *2. #+ users_embedding
        gcn2_items_embedding_1 = (
                torch.sparse.mm(self.item_user_matrix_1, gcn1_users_embedding_1) + gcn1_items_embedding_1.mul(
            self.d_j_train_1))  # *2. #+ items_embedding

        gcn1_users_embedding_0 = (torch.sparse.mm(self.user_item_matrix_0, k_difficulty) + stat_emb.mul(
            self.d_i_train_0))  # *2. + users_embedding
        gcn1_items_embedding_0 = (torch.sparse.mm(self.item_user_matrix_0, stat_emb) + k_difficulty.mul(
            self.d_j_train_0))  # *2. + items_embedding

        gcn2_users_embedding_0 = (
                torch.sparse.mm(self.user_item_matrix_0, gcn1_items_embedding_0) + gcn1_users_embedding_0.mul(
            self.d_i_train_0))  # *2. + users_embedding
        gcn2_items_embedding_0 = (
                torch.sparse.mm(self.item_user_matrix_0, gcn1_users_embedding_0) + gcn1_items_embedding_0.mul(
            self.d_j_train_0))  # *2. + items_embedding

        '''
        graph encoder
        '''
        stat_emb = stat_emb #+ gcn2_users_embedding_1 + gcn2_users_embedding_0
        stat_emb = stat_emb #+ stat_emb_bias
        k_difficulty = k_difficulty #+ gcn2_items_embedding_1 + gcn2_items_embedding_0

        stat_emb = stat_emb[stu_id]
        k_difficulty = k_difficulty[input_exercise]
        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        output_1 = input_x.sum(-1) / input_knowledge_point.sum(-1)
        output_1 = torch.sigmoid(output_1)
        return output_1.view(-1)

    def train_model(self, stu_id, input_exercise, input_knowledge_point, labels):
        output = self.forward(stu_id, input_exercise, input_knowledge_point)
        edu_loss = self.loss_function(output, labels)
        overall = edu_loss
        return overall

    def kl_regulizer(self, mean, std, batch_size):
        '''
        KL term in ELBO loss
        Constraint approximate posterior distribution closer to prior
        '''
        # pdb.set_trace()
        regu_loss = -0.5 * (1 + 2 * std - torch.square(mean) - torch.square(torch.exp(std)))
        return torch.mean(torch.sum(regu_loss, 1, keepdims=True)) / batch_size

    def predict_proficiency_on_concepts(self):
        # before prednet
        stu_emb = self.student_emb.weight
        stat_emb_bias = self.student_emb_bias.weight
        exer_emb = self.exercise_emb.weight
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = self.stat_full(stu_emb * knowledge_emb).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':
            k_difficulty = (exer_emb * knowledge_emb).sum(dim=-1, keepdim=False)  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = self.k_diff_full(exer_emb * knowledge_emb).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)

        gcn1_users_embedding_1 = (torch.sparse.mm(self.user_item_matrix_1, k_difficulty) + stat_emb.mul(
            self.d_i_train_1))  # *2. #+ users_embedding
        gcn1_items_embedding_1 = (torch.sparse.mm(self.item_user_matrix_1, stat_emb) + k_difficulty.mul(
            self.d_j_train_1))  # *2. #+ items_embedding

        gcn2_users_embedding_1 = (
                torch.sparse.mm(self.user_item_matrix_1, gcn1_items_embedding_1) + gcn1_users_embedding_1.mul(
            self.d_i_train_1))  # *2. #+ users_embedding


        gcn1_users_embedding_0 = (torch.sparse.mm(self.user_item_matrix_0, k_difficulty) + stat_emb.mul(
            self.d_i_train_0))  # *2. + users_embedding
        gcn1_items_embedding_0 = (torch.sparse.mm(self.item_user_matrix_0, stat_emb) + k_difficulty.mul(
            self.d_j_train_0))  # *2. + items_embedding

        gcn2_users_embedding_0 = (
                torch.sparse.mm(self.user_item_matrix_0, gcn1_items_embedding_0) + gcn1_users_embedding_0.mul(
            self.d_i_train_0))  # *2. + users_embedding

        stat_emb = stat_emb #+ gcn2_users_embedding_1 + gcn2_users_embedding_0
        stat_emb = stat_emb #+ stat_emb_bias
        return torch.sigmoid(stat_emb)

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
