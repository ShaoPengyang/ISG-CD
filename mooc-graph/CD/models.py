import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import json
from collections import defaultdict
import numpy as np
import math
import copy
import networkx as nx
import pickle
from torch.autograd import Variable
from data_loader import *

def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)    ### real_kernel

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2

class our_adaptive(nn.Module):
    def __init__(self, args, exer_n, student_n, knowledge_n, dim, num_layer, batch_size):
        self.layer_depth = num_layer
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable
        self.batch_size = batch_size
        self.sigma = 0.2
        self.beta = 0.5

        super(our_adaptive, self).__init__()

        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.student_emb_b = nn.Embedding(self.student_n, 1)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.exercise_emb_b = nn.Embedding(self.exer_n, 1)
        self.knowledge_emb = nn.Embedding(self.knowledge_n, self.emb_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # graph learner
        self.activate = nn.ReLU()
        self.edge_bias = nn.Parameter(torch.randn(2))
        self.linear_1_l1 = nn.Linear(in_features=2*self.emb_dim, out_features=self.emb_dim, bias=True)
        self.linear_2_l1 = nn.Linear(in_features=self.emb_dim, out_features=1, bias=True)
        self.linear_1_l0 = nn.Linear(in_features=2 * self.emb_dim, out_features=self.emb_dim, bias=True)
        self.linear_2_l0 = nn.Linear(in_features=self.emb_dim, out_features=1, bias=True)
        self.bce_loss = nn.BCELoss()

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        self.graph_init(args)

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

        self.user_item_matrix_1 = sparse_u_i_1.cuda()
        self.item_user_matrix_1 = sparse_i_u_1.cuda()
        self.user_item_matrix_0 = sparse_u_i_0.cuda()
        self.item_user_matrix_0 = sparse_i_u_0.cuda()

    def chosen_parameter(self, sign):
        if sign == True:
            oppo = False
        else:
            oppo = True
        self.student_emb.weight.requires_grad = sign
        self.student_emb_b.weight.requires_grad = sign
        self.exercise_emb.weight.requires_grad = sign
        self.exercise_emb_b.weight.requires_grad = sign
        self.knowledge_emb.weight.requires_grad = sign
        self.e_discrimination.weight.requires_grad = sign
        self.prednet_full1.weight.requires_grad = sign
        self.prednet_full1.bias.requires_grad = sign
        self.prednet_full2.weight.requires_grad = sign
        self.prednet_full2.bias.requires_grad = sign
        self.prednet_full3.weight.requires_grad = sign
        self.prednet_full3.bias.requires_grad = sign

        self.linear_1_l1.weight.requires_grad = oppo
        self.linear_2_l1.weight.requires_grad = oppo
        self.linear_1_l0.weight.requires_grad = oppo
        self.linear_2_l0.weight.requires_grad = oppo
    
    def graph_learner(self, user_emb, item_emb, detach_choice=False):
        # pdb.set_trace()
        temperature = 0.2
        control = 0.3
        user_ids, item_ids = self.user_item_matrix_1._indices()[0, :], self.user_item_matrix_1._indices()[1, :]
        row_emb = user_emb.weight[user_ids].detach()
        col_emb = item_emb.weight[item_ids].detach()
        out_layer1 = self.activate(self.linear_1_l1(torch.cat([row_emb, col_emb], dim=1)))
        logit = self.linear_2_l1(out_layer1)
        logit = logit.view(-1)
        eps = torch.rand(logit.shape).cuda()
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input)  / temperature
        mask_gate_input = control * torch.sigmoid(mask_gate_input)
        if detach_choice:
            weights = mask_gate_input.detach() + (1-control)
        else:
            weights = mask_gate_input + (1-control)
        masked_user_item_matrix_1 = torch.sparse.FloatTensor(self.user_item_matrix_1._indices(),
                                                             self.user_item_matrix_1._values() * weights)  
        masked_user_item_matrix_1 = masked_user_item_matrix_1.coalesce().cuda()

        item_ids,user_ids = self.item_user_matrix_1._indices()[0, :], self.item_user_matrix_1._indices()[1, :]
        row_emb = user_emb.weight[user_ids].detach()
        col_emb = item_emb.weight[item_ids].detach()
        out_layer1 = self.activate(self.linear_1_l1(torch.cat([row_emb, col_emb], dim=1)))
        logit = self.linear_2_l1(out_layer1)
        logit = logit.view(-1)
        eps = torch.rand(logit.shape).cuda()
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input) / temperature
        mask_gate_input = control * torch.sigmoid(mask_gate_input)
        if detach_choice:
            weights = mask_gate_input.detach() + (1-control)
        else:
            weights = mask_gate_input + (1-control)
        masked_item_user_matrix_1 = torch.sparse.FloatTensor(self.item_user_matrix_1._indices(),
                                                             self.item_user_matrix_1._values() * weights)  
        masked_item_user_matrix_1 = masked_item_user_matrix_1.coalesce().cuda()

        user_ids, item_ids = self.user_item_matrix_0._indices()[0, :], self.user_item_matrix_0._indices()[1, :]
        row_emb = user_emb.weight[user_ids].detach()
        col_emb = item_emb.weight[item_ids].detach()
        out_layer1 = self.activate(self.linear_1_l0(torch.cat([row_emb, col_emb], dim=1)))
        logit = self.linear_2_l0(out_layer1)
        logit = logit.view(-1)
        eps = torch.rand(logit.shape).cuda()
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input) / temperature
        mask_gate_input = control * torch.sigmoid(mask_gate_input)
        if detach_choice:
            weights = mask_gate_input.detach() + (1-control)
        else:
            weights = mask_gate_input + (1-control)
        masked_user_item_matrix_0 = torch.sparse.FloatTensor(self.user_item_matrix_0._indices(),
                                                             self.user_item_matrix_0._values() * weights)  
        masked_user_item_matrix_0 = masked_user_item_matrix_0.coalesce().cuda()

        item_ids, user_ids = self.item_user_matrix_0._indices()[0, :], self.item_user_matrix_0._indices()[1, :]
        row_emb = user_emb.weight[user_ids].detach()
        col_emb = item_emb.weight[item_ids].detach()
        out_layer1 = self.activate(self.linear_1_l0(torch.cat([row_emb, col_emb], dim=1)))
        logit = self.linear_2_l0(out_layer1)
        logit = logit.view(-1)
        eps = torch.rand(logit.shape).cuda()
        mask_gate_input = torch.log(eps) - torch.log(1 - eps)
        mask_gate_input = (logit + mask_gate_input) / temperature
        mask_gate_input = control * torch.sigmoid(mask_gate_input)
        if detach_choice:
            weights = mask_gate_input.detach() + (1-control)
        else:
            weights = mask_gate_input + (1-control)
        masked_item_user_matrix_0 = torch.sparse.FloatTensor(self.item_user_matrix_0._indices(),
                                                             self.item_user_matrix_0._values() * weights) 
        masked_item_user_matrix_0 = masked_item_user_matrix_0.coalesce().cuda()
        return masked_user_item_matrix_1, masked_item_user_matrix_1, masked_user_item_matrix_0, masked_item_user_matrix_0

    def graph_representations(self, user_item_matrix_1, item_user_matrix_1, user_item_matrix_0, item_user_matrix_0):
        stu_emb = self.student_emb.weight 
        exer_emb = self.exercise_emb.weight 
        stu_emb_list = [stu_emb]
        exer_emb_list = [exer_emb]
        for _ in range(self.layer_depth):
            gcn1_users_embedding_1 = torch.sparse.mm(user_item_matrix_1, exer_emb_list[-1])
            gcn1_items_embedding_1 = torch.sparse.mm(item_user_matrix_1, stu_emb_list[-1])
            gcn1_users_embedding_0 = torch.sparse.mm(user_item_matrix_0, exer_emb_list[-1])
            gcn1_items_embedding_0 = torch.sparse.mm(item_user_matrix_0, stu_emb_list[-1])
            tmp_emb_stu = gcn1_users_embedding_1 + gcn1_users_embedding_0
            tmp_emb_exer = gcn1_items_embedding_1 + gcn1_items_embedding_0
            stu_emb_list.append(tmp_emb_stu)
            exer_emb_list.append(tmp_emb_exer)
        stu_emb = torch.stack(stu_emb_list, dim=1)
        stu_emb = torch.mean(stu_emb, dim=1)
        exer_emb = torch.stack(exer_emb_list, dim=1)
        exer_emb = torch.mean(exer_emb, dim=1)
        return stu_emb, exer_emb

    def forward(self, stu_id, input_exercise, input_knowledge_point, labels, detach_choice=False, pre_train=False):
        masked_user_item_matrix_1, masked_item_user_matrix_1, masked_user_item_matrix_0, masked_item_user_matrix_0 = self.graph_learner(
            self.student_emb, self.exercise_emb, detach_choice)
        stu_emb, exer_emb = self.graph_representations(masked_user_item_matrix_1, masked_item_user_matrix_1, \
                                                       masked_user_item_matrix_0, masked_item_user_matrix_0)
        stu_emb_old, exer_emb_old = self.graph_representations(self.user_item_matrix_1, self.item_user_matrix_1, \
                                                               self.user_item_matrix_0, self.item_user_matrix_0)
        ib_loss = self.hsic_graph(stu_id, input_exercise, stu_emb_old, stu_emb, exer_emb_old, exer_emb) * self.beta

        if pre_train:
            stu_emb, exer_emb = stu_emb_old, exer_emb_old

        knowledge_base_emb = self.knowledge_emb.weight
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = knowledge_base_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        stat_emb = (stu_emb * knowledge_emb).sum(dim=-1, keepdim=False)
        stat_emb = stat_emb + self.student_emb_b.weight

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = knowledge_base_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        k_difficulty = (exer_emb * knowledge_emb).sum(dim=-1, keepdim=False)
        k_difficulty = k_difficulty + self.exercise_emb_b.weight

        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))
        stat_emb = torch.sigmoid(stat_emb[stu_id])
        k_difficulty = torch.sigmoid(k_difficulty[input_exercise])
        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        input_x = (torch.tanh(self.prednet_full1(input_x)))
        input_x = (torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        edu_loss = self.bce_loss(output_1.view(-1), labels)
        return edu_loss, ib_loss

    def return_output(self, stu_id, input_exercise, input_knowledge_point):
        masked_user_item_matrix_1, masked_item_user_matrix_1, masked_user_item_matrix_0, masked_item_user_matrix_0 = self.graph_learner(
            self.student_emb, self.exercise_emb)
        stu_emb, exer_emb = self.graph_representations(masked_user_item_matrix_1, masked_item_user_matrix_1, \
                                                       masked_user_item_matrix_0, masked_item_user_matrix_0)

        knowledge_base_emb = self.knowledge_emb.weight
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = knowledge_base_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        stat_emb = (stu_emb * knowledge_emb).sum(dim=-1, keepdim=False)
        stat_emb = stat_emb + self.student_emb_b.weight

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = knowledge_base_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        k_difficulty = (exer_emb * knowledge_emb).sum(dim=-1, keepdim=False)
        k_difficulty = k_difficulty + self.exercise_emb_b.weight
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        stat_emb = torch.sigmoid(stat_emb[stu_id])
        k_difficulty = torch.sigmoid(k_difficulty[input_exercise])
        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        input_x = (torch.tanh(self.prednet_full1(input_x)))
        input_x = (torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        return output_1.view(-1)

    def hsic_graph(self, users, items, user_emb_old, user_emb, item_emb_old, item_emb):
        ### user part ###
        users = torch.unique(users)
        items = torch.unique(items)
        input_x = user_emb_old[users]
        input_y = user_emb[users]
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, self.batch_size)
        ### item part ###
        input_i = item_emb_old[items]
        input_j = item_emb[items]
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, self.batch_size)
        loss = loss_user + loss_item
        return loss

    def predict_proficiency_on_concepts(self):
        masked_user_item_matrix_1, masked_item_user_matrix_1, masked_user_item_matrix_0, masked_item_user_matrix_0 = self.graph_learner(
            self.student_emb, self.exercise_emb)
        stu_emb, exer_emb = self.graph_representations(masked_user_item_matrix_1, masked_item_user_matrix_1, \
                                                       masked_user_item_matrix_0, masked_item_user_matrix_0)
        knowledge_base_emb = self.knowledge_emb.weight
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = knowledge_base_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        stat_emb = (stu_emb * knowledge_emb).sum(dim=-1, keepdim=False)
        return torch.sigmoid(stat_emb+self.student_emb_b.weight)




def irt2pl(theta, a, b, F=np):
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))



def irf(theta, a, b, c, D=1.702, *, F=np):
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))


irt3pl = irf

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
        # get knowledge proficiency
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
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        stat_emb = stat_emb[stu_id]
        k_difficulty = k_difficulty[input_exercise]

        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        # f = input_x[input_knowledge_point == 1]
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
