import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path as osp
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import EduData
from model_DAG4junyi import *
from utils import *
from torch.utils.data import DataLoader
import time
import math
import cdt
import pdb
from cdt.data import load_dataset
import networkx as nx
# 导入的数据看起来应该满足一个pd，一个nx的有向图
# s_data, s_graph = load_dataset('sachs')
# pdb.set_trace()

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)

def train(args):
    ### Education data pre-processing
    True_DAG = pd.read_csv('../data/junyi/hier.csv')

    best_auc = 0
    train_dataset = EduData(type='train')
    train_dataset.load_data()
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)
    test_dataset = EduData(type='predict')
    test_dataset.load_data()
    test_loader = DataLoader(test_dataset, batch_size=8192, shuffle=False)
    print("len of training dataset is: " + str(len(train_dataset)))
    print("len of test dataset is: " + str(len(test_dataset)))

    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')

    loss_function = nn.BCELoss(reduction='none')
    net = NCDM_dag(args.knowledge_n, args.exer_n,args.student_n, True_DAG)
    # net = NCDM(args, args.knowledge_n, args.exer_n, args.student_n)
    # net = KaNCD_dag(args, args.exer_n, args.student_n, args.knowledge_n, 'gmf', 64, True_DAG)
    net = net.to(device)
    optimizer_net = optim.Adam(net.parameters(), lr=0.005)
    # optimizer_net = optim.Adam(net.parameters(), lr=0.0005)
    for epoch in range(10):
        # 为了让模型能见过所有样本，必须把学习率调低
        net.train()
        running_loss = []
        for idx, (input_stu_ids, input_exer_ids, kid, labels) in enumerate(train_loader):
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            kid = kid.cuda()
            labels = labels.cuda().float()
            optimizer_net.zero_grad()
            output = net.forward(input_stu_ids, input_exer_ids,kid, epoch)
            edu_loss = loss_function(output, labels)
            edu_loss = torch.sum(edu_loss * 1)
            edu_loss.backward(retain_graph=True)
            optimizer_net.step()
            running_loss.append(edu_loss.item())

        predict(args, net,  test_loader, epoch, best_auc)
        result, last_w = net.predict_results_dag()
        print(result)
        print(last_w)

def predict(args, net, test_loader, epoch, best_auc):
    net.eval()
    with torch.no_grad():
        correct_count, exer_count = 0, 0
        pred_all, label_all = [], []
        for input_stu_ids, input_exer_ids, input_knowledge_embs, labels in test_loader:
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            labels = labels.cuda()
            input_knowledge_embs = torch.cuda.LongTensor(input_knowledge_embs)
            output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs, epoch)
            output = output.view(-1)

            correct_count += ((output >= 0.5) == labels).sum()
            # compute accuracy
            # for i in range(len(labels)):
            #     if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
            #         correct_count += 1
            exer_count += len(labels)
            pred_all += output.to(torch.device('cpu')).tolist()
            label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    if auc > best_auc:
        torch.save(net.state_dict(),"best_net.pt")
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    train(args)
