import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

    net = our_adaptive(args, args.exer_n, args.student_n, args.knowledge_n, 64, 2, 0.05)
    net = net.to(device)

    optimizer_net = optim.Adam(net.parameters(), lr=0.0005)

    # 18
    for epoch in range(100):
        if epoch+1 > 2:
            print("adaptively dividing links process")
            with torch.no_grad():
                users, items, labels_list, predicted_scores = [], [], [], []
                for idx, (input_stu_ids, input_exer_ids, kid, labels) in enumerate(train_loader):
                    input_stu_ids = input_stu_ids.cuda()
                    input_exer_ids = input_exer_ids.cuda()
                    kid = kid.cuda().long()
                    labels = labels.cuda().float()
                    output_1 = net.forward(input_stu_ids, input_exer_ids, kid)
                    users.extend(input_stu_ids.to(torch.device('cpu')).tolist())
                    items.extend(input_exer_ids.to(torch.device('cpu')).tolist())
                    labels_list.extend(labels.to(torch.device('cpu')).tolist())
                    predicted_scores.extend(output_1.to(torch.device('cpu')).tolist())

            all_lists = [users, items, labels_list, predicted_scores]
            all_lists = np.array(all_lists).T
            net.graph_update(args, all_lists)
        net.train()
        running_loss = []
        for idx, (input_stu_ids, input_exer_ids, kid, labels) in enumerate(train_loader):
            input_stu_ids = input_stu_ids.cuda()
            input_exer_ids = input_exer_ids.cuda()
            kid = kid.cuda().long()
            labels = labels.cuda().float()
            optimizer_net.zero_grad()
            output = net.forward(input_stu_ids, input_exer_ids,kid)
            edu_loss = loss_function(output, labels)
            edu_loss = torch.sum(edu_loss * 1)
            edu_loss.backward(retain_graph=True)
            optimizer_net.step()
            running_loss.append(edu_loss.item())

        predict(args, net,  test_loader, epoch, best_auc)



def predict(args, net, test_loader, epoch, best_auc):
    net.eval()
    users, items, know_ids, predicted_scores, predicted_thetas = [], [], [], [], []
    with torch.no_grad():
        prof = net.predict_proficiency_on_concepts().to(torch.device('cpu')).numpy()
        print(np.mean(prof))
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
            output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            # output = net.forward(input_stu_ids, input_exer_ids)
            output = output.view(-1)
            predicted_scores.extend(labels.to(torch.device('cpu')).tolist())
            correct_count += ((output >= 0.5) == labels).sum()
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
    DOA_R = DOA['doa']
    # DOA_R = 0
    print('[epoch= %d], accuracy= %f, rmse= %f, auc= %f doa= %f' % (epoch+1, accuracy, rmse, auc, DOA_R))

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    train(args)
