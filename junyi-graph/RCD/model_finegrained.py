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


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


train_data_json = '../data/junyi/train_set_ood2-HCD.json'
test_data_json = '../data/junyi/test_set_ood2-HCD.json'


class KaNCD_RGCN_fine1(nn.Module):
    def __init__(self, args, exer_n, student_n, knowledge_n, mf_type, dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable

        super(KaNCD_RGCN_fine1, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.student_emb_bias = nn.Embedding(self.student_n, 1)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

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

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)
        self.loss_function = nn.BCELoss()
        self.graph_init(args)
        self.update = False

    def graph_init(self, args):
        with open(train_data_json, encoding='utf8') as i_f:
            data = json.load(i_f)
        self.mask = torch.zeros((args.student_n, args.exer_n))
        interaction0 = torch.zeros((args.student_n, args.exer_n))
        interaction1 = torch.zeros((args.student_n, args.exer_n))
        for idx, log in enumerate(data):
            u_id = log['user_id']
            i_id = log['exer_id']
            self.mask[u_id][i_id] = 1
            if log['score'] == 1:
                interaction1[u_id][i_id] = 1
            elif log['score'] == 0:
                interaction0[u_id][i_id] = 1
            else:
                assert False, 'rating must be 1 or 0.'
        # student-exercise. Each student representation should consist of his interacted items.
        # So, 1/item num is a ok weight for a user
        tmp_sum = interaction0.sum(dim=1).view(-1, 1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_ui0 = interaction0 / tmp_sum
        sparse_ui_0 = tmp_ui0.to_sparse()

        tmp_sum = interaction0.sum(dim=0).view(1, -1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_iu0 = (interaction1 / tmp_sum).t()
        sparse_iu_0 = tmp_iu0.to_sparse()

        tmp_sum = interaction1.sum(dim=1).view(-1, 1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_ui1 = interaction1 / tmp_sum
        sparse_ui_1 = tmp_ui1.to_sparse()

        tmp_sum = interaction1.sum(dim=0).view(1, -1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_iu1 = (interaction1 / tmp_sum).t()
        sparse_iu_1 = tmp_iu1.to_sparse()

        self.user_item_matrix_1 = sparse_ui_1.cuda()
        self.item_user_matrix_1 = sparse_iu_1.cuda()
        self.user_item_matrix_0 = sparse_ui_0.cuda()
        self.item_user_matrix_0 = sparse_iu_0.cuda()

    def graph_update(self, args, predicted_results):
        '''
        :param predicted_results: np.array [users, items, labels, predicted_scores]
        :return:
        '''
        pos_interaction_true = torch.zeros((args.student_n, args.exer_n))
        pos_interaction_false = torch.zeros((args.student_n, args.exer_n))
        neg_interaction_true = torch.zeros((args.student_n, args.exer_n))
        neg_interaction_false = torch.zeros((args.student_n, args.exer_n))

        pos_interactions = predicted_results[np.where(predicted_results[:, 2] == 1)]
        neg_interactions = predicted_results[np.where(predicted_results[:, 2] == 0)]

        threshold = 0.7
        pos_interactions_real = pos_interactions[np.where(pos_interactions[:, 3] > threshold)]
        pos_interactions_close = pos_interactions[np.where(pos_interactions[:, 3] <= threshold)]
        threshold2 = 0.55  # 0.55
        neg_interactions_real = pos_interactions[np.where(neg_interactions[:, 3] < threshold2)]
        neg_interactions_close = pos_interactions[np.where(neg_interactions[:, 3] >= threshold2)]

        print(len(pos_interactions_real))
        print(len(pos_interactions_close))
        print(len(neg_interactions_real))
        print(len(neg_interactions_close))
        if len(pos_interactions_real) > 0:
            for value in pos_interactions_real:
                user = int(value[0])
                item = int(value[1])
                pos_interaction_true[user][item] = 1
        else:
            user = args.student_n - 1
            item = args.exer_n - 1
            pos_interaction_true[user][item] = 0.001

        if len(pos_interactions_close) > 0:
            for value in pos_interactions_close:
                user = int(value[0])
                item = int(value[1])
                pos_interaction_false[user][item] = 1
        else:
            user = args.student_n - 1
            item = args.exer_n - 1
            pos_interaction_false[user][item] = 0.001

        if len(neg_interactions_real) > 0:
            for value in neg_interactions_real:
                user = int(value[0])
                item = int(value[1])
                neg_interaction_true[user][item] = 1
        else:
            user = args.student_n - 1
            item = args.exer_n - 1
            neg_interaction_true[user][item] = 0.001

        if len(neg_interactions_close) > 0:
            for value in neg_interactions_close:
                user = int(value[0])
                item = int(value[1])
                neg_interaction_false[user][item] = 1
        else:
            user = args.student_n - 1
            item = args.exer_n - 1
            neg_interaction_false[user][item] = 0.001

        # user-item
        tmp_sum = pos_interaction_true.sum(dim=1).view(-1, 1)
        # avoid nan
        tmp_sum[tmp_sum == 0] = 1
        tmp_ui_postrue = pos_interaction_true / tmp_sum
        sparse_ui_postrue = tmp_ui_postrue.to_sparse()

        tmp_sum = pos_interaction_false.sum(dim=1).view(-1, 1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_ui_posfalse = pos_interaction_false / tmp_sum
        sparse_ui_posfalse = tmp_ui_posfalse.to_sparse()

        tmp_sum = neg_interaction_true.sum(dim=1).view(-1, 1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_ui_negtrue = neg_interaction_true / tmp_sum
        sparse_ui_negtrue = tmp_ui_negtrue.to_sparse()

        tmp_sum = neg_interaction_false.sum(dim=1).view(-1, 1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_ui_negfalse = neg_interaction_false / tmp_sum
        sparse_ui_negfalse = tmp_ui_negfalse.to_sparse()

        # item-user
        tmp_sum = pos_interaction_true.sum(dim=0).view(1, -1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_iu_postrue = (pos_interaction_true / tmp_sum).t()
        sparse_iu_postrue = tmp_iu_postrue.to_sparse()

        tmp_sum = pos_interaction_false.sum(dim=0).view(1, -1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_iu_posfalse = (pos_interaction_false / tmp_sum).t()
        sparse_iu_posfalse = tmp_iu_posfalse.to_sparse()

        tmp_sum = neg_interaction_true.sum(dim=0).view(1, -1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_iu_negtrue = (neg_interaction_true / tmp_sum).t()
        sparse_iu_negtrue = tmp_iu_negtrue.to_sparse()

        tmp_sum = neg_interaction_false.sum(dim=0).view(1, -1)
        tmp_sum[tmp_sum == 0] = 1
        tmp_iu_negfalse = (neg_interaction_false / tmp_sum).t()
        sparse_iu_negfalse = tmp_iu_negfalse.to_sparse()

        self.user_item_matrix_postrue = sparse_ui_postrue.cuda()
        self.user_item_matrix_posfalse = sparse_ui_posfalse.cuda()
        self.user_item_matrix_negtrue = sparse_ui_negtrue.cuda()
        self.user_item_matrix_negfalse = sparse_ui_negfalse.cuda()

        self.item_user_matrix_postrue = sparse_iu_postrue.cuda()
        self.item_user_matrix_posfalse = sparse_iu_posfalse.cuda()
        self.item_user_matrix_negtrue = sparse_iu_negtrue.cuda()
        self.item_user_matrix_negfalse = sparse_iu_negfalse.cuda()
        self.update = True

    def graph_representations(self):
        stu_emb = self.student_emb.weight
        exer_emb = self.exercise_emb.weight
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
        if not self.update:
            gcn1_users_embedding_1 = torch.sparse.mm(self.user_item_matrix_1, k_difficulty)
            gcn1_items_embedding_1 = torch.sparse.mm(self.item_user_matrix_1, stat_emb)

            gcn2_users_embedding_1 = torch.sparse.mm(self.user_item_matrix_1, gcn1_items_embedding_1)
            gcn2_items_embedding_1 = torch.sparse.mm(self.item_user_matrix_1, gcn1_users_embedding_1)

            gcn1_users_embedding_0 = torch.sparse.mm(self.user_item_matrix_0, k_difficulty)  # *2. + users_embedding
            gcn1_items_embedding_0 = torch.sparse.mm(self.item_user_matrix_0, stat_emb)  # *2. + items_embedding

            gcn2_users_embedding_0 = torch.sparse.mm(self.user_item_matrix_0,
                                                     gcn1_items_embedding_0)  # *2. + users_embedding
            gcn2_items_embedding_0 = torch.sparse.mm(self.item_user_matrix_0,
                                                     gcn1_users_embedding_0)  # *2. + items_embedding
            stat_emb = stat_emb + gcn2_users_embedding_1 + gcn2_users_embedding_0
            # stat_emb = stat_emb + stat_emb_bias
            k_difficulty = k_difficulty + gcn2_items_embedding_1 + gcn2_items_embedding_0
        else:
            gcn1_users_embedding_postrue = torch.sparse.mm(self.user_item_matrix_postrue, k_difficulty)
            gcn1_items_embedding_postrue = torch.sparse.mm(self.item_user_matrix_postrue, stat_emb)
            gcn2_users_embedding_postrue = torch.sparse.mm(self.user_item_matrix_postrue, gcn1_items_embedding_postrue)
            gcn2_items_embedding_postrue = torch.sparse.mm(self.item_user_matrix_postrue, gcn1_users_embedding_postrue)

            gcn1_users_embedding_posfalse = torch.sparse.mm(self.user_item_matrix_posfalse, k_difficulty)
            gcn1_items_embedding_posfalse = torch.sparse.mm(self.item_user_matrix_posfalse, stat_emb)
            gcn2_users_embedding_posfalse = torch.sparse.mm(self.user_item_matrix_posfalse,
                                                            gcn1_items_embedding_posfalse)
            gcn2_items_embedding_posfalse = torch.sparse.mm(self.item_user_matrix_posfalse,
                                                            gcn1_users_embedding_posfalse)

            gcn1_users_embedding_negtrue = torch.sparse.mm(self.user_item_matrix_negtrue, k_difficulty)
            gcn1_items_embedding_negtrue = torch.sparse.mm(self.item_user_matrix_negtrue, stat_emb)
            gcn2_users_embedding_negtrue = torch.sparse.mm(self.user_item_matrix_negtrue, gcn1_items_embedding_negtrue)
            gcn2_items_embedding_negtrue = torch.sparse.mm(self.item_user_matrix_negtrue, gcn1_users_embedding_negtrue)

            gcn1_users_embedding_negfalse = torch.sparse.mm(self.user_item_matrix_negfalse, k_difficulty)
            gcn1_items_embedding_negfalse = torch.sparse.mm(self.item_user_matrix_negfalse, stat_emb)
            gcn2_users_embedding_negfalse = torch.sparse.mm(self.user_item_matrix_negfalse,
                                                            gcn1_items_embedding_negfalse)
            gcn2_items_embedding_negfalse = torch.sparse.mm(self.item_user_matrix_negfalse,
                                                            gcn1_users_embedding_negfalse)

            stat_emb = stat_emb + gcn2_users_embedding_postrue + gcn2_users_embedding_posfalse \
                       + gcn2_users_embedding_negtrue + gcn2_users_embedding_negfalse
            k_difficulty = k_difficulty + gcn2_items_embedding_postrue + gcn2_items_embedding_posfalse + \
                           gcn2_items_embedding_negtrue + gcn2_items_embedding_negfalse
        return stat_emb, k_difficulty

    def forward(self, stu_id, input_exercise, input_knowledge_point, labels):
        stat_emb, k_difficulty = self.graph_representations()
        # cognitive diagnosis loss
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))
        stat_emb = stat_emb[stu_id]
        k_difficulty = k_difficulty[input_exercise]
        input_x = input_knowledge_point * (stat_emb - k_difficulty) * e_discrimination
        output_1 = input_x.sum(-1) / input_knowledge_point.sum(-1)
        output_1 = torch.sigmoid(output_1)
        cd_loss = self.loss_function(output_1.view(-1).float(), labels.float())
        return output_1, cd_loss

    def predict_proficiency_on_concepts(self):
        stat_emb, k_difficulty = self.graph_representations()
        return torch.sigmoid(stat_emb)


