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

train_data_json = '../data/coarse/train_set1.npy'
test_data_json = '../data/coarse/test_set1.npy'
eval_data_json = '../data/coarse/eval_set1.npy'
item2knowledge_path = "../data/coarse/item2knowledge.npy"
item2knowledge = np.load(item2knowledge_path, allow_pickle = True).item()

def obtain_adjency_matrix(args):
    data = np.load(train_data_json, allow_pickle=True)
    train_data_user_score1,train_data_user_score0 = defaultdict(set), defaultdict(set)
    train_data_item_score1,train_data_item_score0 = defaultdict(set), defaultdict(set)
    for idx, log in enumerate(data):
        u_id = log[0] -1
        i_id = log[1] -1
        if log[2] == 1:
            train_data_user_score1[u_id].add(int(i_id))
            train_data_item_score1[int(i_id)].add(u_id)
        elif log[2] == 0:
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
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
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
        elif type == 'eval':
            self.data_file = eval_data_json
            self.type = 'eval'
        else:
            assert False, 'type can only be selected from train or predict'
        self.data = np.load(self.data_file, allow_pickle=True)
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
        self.data_len = self.data.shape[0]
        self.knowledge = torch.zeros((self.exercise_dim, self.knowledge_dim)).cuda()

    def __len__(self):
        return  self.data_len

    def __getitem__(self, idx):
        u_id = self.data[idx][0] - 1
        i_id = self.data[idx][1] - 1
        label = self.data[idx][2]
        xxx = torch.zeros(self.knowledge_dim)
        xxx[item2knowledge[self.data[idx][1]]-1] = 1
        k_id = xxx
        return u_id, i_id, k_id, label

class EduData_DA(data.Dataset):
    def __init__(self, type='train'):
        super(EduData_DA, self).__init__()
        if type == 'train':
            self.data_file = train_data_json
            self.type = 'train'
        elif type == 'predict':
            self.data_file = test_data_json
            self.type = 'predict'
        else:
            assert False, 'type can only be selected from train or predict'
        self.config_file = 'config.txt'
        with open(self.config_file) as i_f:
            i_f.readline()
            student_n, exercise_n, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.student_dim = int(student_n)
        self.exercise_dim = int(exercise_n)

    def load_data(self, random_sample=None, augmention_labels=None, corresponding_concept=None):
        '''
        if first load, use comment part.
        :return:
        '''
        self.knowledge = torch.zeros((self.exercise_dim, self.knowledge_dim)).cuda()

        self.dataset = []
        self.k_ids = []
        # data = np.load(self.data_file, allow_pickle=True)
        # for idx, log in enumerate(data):
        #     u_id = data[idx][0] - 1
        #     e_id = data[idx][1] - 1
        #     y = data[idx][2]
        #     weight = 1
        #     self.dataset.append([u_id, e_id, y, weight])
        if random_sample is not None:
            a_u_ids = random_sample[:,0]
            a_e_ids = random_sample[:, 1]
            for idx, label in enumerate(augmention_labels):
                if label == 1 or label == 0:
                    weight = 1
                    self.dataset.append([int(a_u_ids[idx]), int(a_e_ids[idx]), int(label), int(weight)])
        print(len(self.dataset))
        self.data_len = len(self.dataset)

    def __len__(self):
        return  self.data_len

    def __getitem__(self, idx):
        try:
            u_id = self.dataset[idx][0]
            # from zero
            i_id = self.dataset[idx][1]
            label = self.dataset[idx][2]
            xxx = torch.zeros(self.knowledge_dim)
            # item2knowledge key is from 1
            xxx[item2knowledge[self.dataset[idx][1] + 1] - 1] = 1
            k_id = xxx
            weight = self.dataset[idx][3]
        except:
            pdb.set_trace()
        return u_id, i_id, k_id, label, weight


def generate_random_sample(epoch, DeficientConceptDict, ConceptMapExercise, ExerciseMapConcept, max_number=10):
    '''
    :param DeficientConceptDict: where needs to perform data augmentation
    :param ConceptMapExercise: concept:{exercise1, exercise2, ... , exercise S}
    :param max_number: maxed added number for each {student,concept} pair.
    :return: random_sample (candidates)
    '''
    np.random.seed(epoch*3+1)
    random.seed(epoch*3+1)

    random_sample = []
    corresponding_concept_vector = []
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exercise_n, knowledge_n = i_f.readline().split(',')
    student_n, exercise_n, knowledge_n = int(student_n), int(exercise_n), int(knowledge_n)

    for student, interactions in DeficientConceptDict.items():
        for concept, exercises in interactions.items():
            all_exercises_set = ConceptMapExercise[concept]
            done_exercises_set = set(exercises)
            differences = np.array(list(all_exercises_set - done_exercises_set))
            sample_number = max_number - len(done_exercises_set)
            if (sample_number > 0) and (sample_number < len(differences)):
                add_part = np.random.choice(differences, size=sample_number, replace=False)
            else:
                add_part = differences

        # pdb.set_trace()
        assert len(add_part) == len(set(list(add_part))), "repeatable elements!!!"
        for exercise in add_part:
            knowledge_emb = [0.] * knowledge_n
            for knowledge_code in ExerciseMapConcept[exercise]:
                knowledge_emb[knowledge_code] = 1.0

            random_sample.append([student, exercise])
            corresponding_concept_vector.append(knowledge_emb)
    random_sample = np.array(random_sample)

    return random_sample, corresponding_concept_vector


def PreprocessData(StandardLength=5, user_num=1, exer_num=1, concept_num=1):
    data = np.load(train_data_json, allow_pickle=True)
    StudentExerciseMatrix, StuentConceptTimes = np.zeros((user_num,exer_num)), np.zeros((user_num,concept_num))
    # ExerciseMapConcept = defaultdict(set)
    for idx, log in enumerate(data):
        u_id = log[0] - 1
        e_id = log[1] - 1
        label = log[2]
        knowledge_id = item2knowledge[log[1]] - 1
        StuentConceptTimes[u_id][knowledge_id] += 1
        # 1 0 turn 0 1 -1 for similarity calculation
        StudentExerciseMatrix[u_id][e_id] = (label*2-1)

    lts_data_tensor = torch.cuda.FloatTensor(StudentExerciseMatrix)
    cos_sim = sim_matrix(lts_data_tensor,lts_data_tensor)

    # An operation.
    cos_sim[cos_sim < 0.1] = 0
    cos_sim[cos_sim > 0.999] = 1
    cos_sim = cos_sim - torch.diag_embed(torch.diag(cos_sim))
    StudentSimilarityMatrix = cos_sim.detach().cpu().numpy()
    from copy import deepcopy
    StuentConceptTimesv2 = deepcopy(StuentConceptTimes)
    StuentConceptTimesv2[StuentConceptTimesv2 < StandardLength] = 0
    StuentConceptTimesv2[StuentConceptTimesv2 >= StandardLength] = 1
    return StudentSimilarityMatrix, StuentConceptTimesv2

def sim_matrix(a, b, eps=1e-10):
    """
    added eps for numerical stability
    a: M*E,
    a.norm(dim=1): M
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.sparse.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
