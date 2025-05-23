import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path as osp
import os
import networkx as nx
import json
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
from data_loader import EduData
from models import *
from utils import *
from torch.utils.data import DataLoader
import time
import math
import pdb

torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)

def train(args):
    best_auc = 0
    batch_size = 8192
    train_dataset = EduData(type='train')
    train_dataset.load_data()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = EduData(type='predict')
    test_dataset.load_data()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    eval_dataset = EduData(type='eval')
    eval_dataset.load_data()
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    print("len of training dataset is: " + str(len(train_dataset)))
    print("len of test dataset is: " + str(len(test_dataset)))
    net = our_adaptive(args, args.exer_n, args.student_n, args.knowledge_n, 128, 4, batch_size)
    net = net.cuda()

    optimizer_net = optim.Adam(net.parameters(), lr=0.0005)
    for epoch in range(5):
        net.train()
        for idx, (input_stu_ids, input_exer_ids, kid, labels) in enumerate(train_loader):
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            kid = kid.cuda().long()
            labels = labels.cuda().float()
            optimizer_net.zero_grad()
            edu_loss, ib_loss = net.forward(input_stu_ids, input_exer_ids, kid, labels, detach_choice=True, pre_train=False)
            loss = edu_loss
            loss.backward()
            optimizer_net.step()

    for epoch in range(30):
        net.train()
        net.chosen_parameter(False)
        for idx, (input_stu_ids, input_exer_ids, kid, labels) in enumerate(train_loader):
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            kid = kid.cuda().long()
            labels = labels.cuda().float()
            optimizer_net.zero_grad()
            edu_loss, ib_loss = net.forward(input_stu_ids, input_exer_ids, kid, labels)
            loss = edu_loss + ib_loss
            loss.backward()
            optimizer_net.step()

        net.chosen_parameter(True)
        for idx, (input_stu_ids, input_exer_ids, kid, labels) in enumerate(train_loader):
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            kid = kid.cuda().long()
            labels = labels.cuda().float()
            optimizer_net.zero_grad()
            edu_loss, ib_loss = net.forward(input_stu_ids, input_exer_ids, kid, labels, detach_choice=True)
            loss = edu_loss
            loss.backward()
            optimizer_net.step()

        print("epoch:" + str(epoch))
        predict(args, net, eval_loader, epoch, best_auc)
        predict(args, net, test_loader, epoch, best_auc)


def predict(args, net, test_loader, epoch, best_auc):
    net.eval()
    users, items, know_ids, predicted_scores, predicted_thetas = [], [], [], [], []
    with torch.no_grad():
        prof = net.predict_proficiency_on_concepts().to(torch.device('cpu')).numpy()
        # print(np.mean(prof))
        correct_count, exer_count = 0, 0
        pred_all, label_all = [], []
        for input_stu_ids, input_exer_ids, input_knowledge_embs, labels in test_loader:
            for _ in input_stu_ids.cpu().numpy():
                predicted_thetas.append(list(prof[_]))
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            labels = labels.cuda()
            input_knowledge_embs = input_knowledge_embs.cuda()
            users.extend(input_stu_ids.to(torch.device('cpu')).tolist())
            items.extend(input_exer_ids.to(torch.device('cpu')).tolist())
            know_ids.extend(input_knowledge_embs.to(torch.device('cpu')).tolist())
            output = net.return_output(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output = output.view(-1)
            predicted_scores.extend(labels.to(torch.device('cpu')).tolist())
            correct_count += ((output >= 0.5) == labels).sum().item()
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
    # print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch, accuracy, rmse, auc))
    DOA =  doa_report(users, items, know_ids, predicted_scores, predicted_thetas)
    print('[epoch= %d], accuracy= %f, rmse= %f, auc= %f doa= %f' % (epoch+1, accuracy, rmse, auc, DOA['doa']))

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    print(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    train(args)
