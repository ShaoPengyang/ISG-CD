import json
import random
from collections import defaultdict
import pdb
import pandas as pd
import numpy as np
import os.path as osp
import networkx as nx


min_log = 0 #15

def divide_data_OOD_HCD():
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set and test_set (0.8:0.2)
    :return:
    '''

    train_data = pd.read_csv('../data/junyi/train_0.8_0.2.csv', encoding='utf-8', low_memory=True)
    test_data = pd.read_csv('../data/junyi/test_0.8_0.2.csv', encoding='utf-8', low_memory=True)
    train_set = []
    test_set = []
    for index, record in train_data.iterrows():
        student = record['user_id']
        exer = record['exer_id']
        score = record['score']
        train_set.append({'user_id': int(student), 'exer_id': int(exer), 'score': int(score),
                         'knowledge_code': int(exer)})
    for index, record in test_data.iterrows():
        student = record['user_id']
        exer = record['exer_id']
        score = record['score']
        test_set.append({'user_id': int(student), 'exer_id': int(exer), 'score': int(score),
                         'knowledge_code': int(exer)})
    print(len(train_set))
    print(len(test_set))
    pdb.set_trace()
    with open('../data/junyi/train_set_ood2-HCD.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('../data/junyi/test_set_ood2-HCD.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set


def divide_data_OOD():
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set and test_set (0.8:0.2)
    :return:
    '''
    DATA_PATH = '../../junyi-origin/'

    all_data = np.load(osp.join(DATA_PATH, 'log.npy'),allow_pickle=True).item()
    id2name = np.load(osp.join(DATA_PATH, 'id2name.npy'), allow_pickle=True).item()
    # dependency = pd.read_csv(osp.join(DATA_PATH, 'junyi_Exercise_table.npy'), encoding='utf-8', low_memory=True)
    item_times = defaultdict(int)
    user_number = 6000
    for student, current_student_log in all_data.items():
        if int(student) < user_number:
            for exercise, record in current_student_log.items():
                item_times[exercise] += 1
    old_id = dict()
    for row,index in item_times.items():
        if index > 5:
            old_id[row] = 1
    item_old2newid = dict()
    for i, (k, v) in enumerate(old_id.items()):
        item_old2newid[k] = i

    np.save("../data/item_old2newid.npy", item_old2newid)
    max_user = 0
    max_item = 0
    all_set = []
    for student, current_student_log in all_data.items():
        if int(student) < user_number:
            if int(student) > max_user:
                max_user = int(student)
            for exercise, record in current_student_log.items():
                if exercise in item_old2newid.keys():
                    new_e_id = item_old2newid[exercise]
                    if new_e_id > max_item:
                        max_item = new_e_id
                    all_set.append(
                        {'user_id': int(student), 'exer_id': new_e_id, 'score': int(record[0]),
                         'knowledge_code': int(exercise)})

    random.shuffle(all_set)
    length = int(0.8*len(all_set))
    train_set = all_set[:length]
    test_set = all_set[length:]
    print(len(train_set))
    print(len(test_set))
    print(max_item)
    print(max_user)
    pdb.set_trace()
    with open('../data/junyi/train_set_ood2.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open('../data/junyi/test_set_ood2.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set


if __name__ == '__main__':
    divide_data_OOD_HCD()
