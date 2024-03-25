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

if __name__ == '__main__':
    divide_data_OOD_HCD()
