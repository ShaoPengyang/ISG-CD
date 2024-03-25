import json
import random
from collections import defaultdict
import pdb
import pandas as pd
import numpy as np
import os.path as osp
import networkx as nx
import tqdm


min_log = 0 #15

def divide_data():
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set and test_set (0.8:0.2)
    :return:
    '''
    all_data_json = "../data/coarse/student-problem-coarse.json"
    problem_json = "../data/coarse/problem.json"
    # data = pd.read_csv(all_data_json)
    with open(all_data_json, 'r', encoding='utf-8') as f_in:
        json_data = json.loads(f_in.read())

    df_nested_list = pd.json_normalize(json_data, record_path=['seq'])
    raw_question = df_nested_list.problem_id.unique().tolist()
    num_skill = len(raw_question)
    # problem map: (start from 1)
    map_problems = {p: i + 1 for i, p in enumerate(raw_question)}
    num_questions = num_skill

    with open(problem_json, 'r', encoding='utf-8')as f_in:
        problem_data = []
        for line in f_in.readlines():
            dic = json.loads(line)
            problem_data.append(dic)

    df_problem_detail = pd.json_normalize(problem_data)
    df_problem_detail_sub = df_problem_detail[df_problem_detail.problem_id.isin(raw_question)]

    raw_concept = set()
    for c in df_problem_detail_sub.exercise_id.tolist():
        raw_concept.add(c)
    num_concept = len(raw_concept)
    # concept map: (start from 1)
    map_concepts = {c: i + 1 for i, c in enumerate(raw_concept)}

    item2knowledge = {}
    for index, row in tqdm.tqdm(df_problem_detail_sub.iterrows()):
        problem_id = row['problem_id']
        concept = row['exercise_id']
        item2knowledge[map_problems[problem_id]] = map_concepts[concept]

    # saved
    item_df = pd.DataFrame({'item_id': item2knowledge.keys(), 'knowledge_code': item2knowledge.values()})
    # item_df.to_csv(args.saved_item_dir, sep=',', index=False, header=True)

    # 4. build user map
    raw_users = df_nested_list.user_id.unique().tolist()
    num_users = len(raw_users)
    # users map: (start from 1)
    map_users = {u: i + 1 for i, u in enumerate(raw_users)}
    # args.num_questions = num_users

    # 5. build dataset
    dataset = df_nested_list[['user_id', 'problem_id', 'is_correct']]
    dataset['user_id'] = dataset['user_id'].map(map_users)
    dataset['problem_id'] = dataset['problem_id'].map(map_problems)

    dataset = dataset.sample(frac=1)

    # all_set_dict = defaultdict(set)
    # for index, row in dataset.iterrows():
    #     student = row['user_id']
    #     exer = row['problem_id']
    #     score = row['is_correct']
    #     all_set_dict[student].add(item2knowledge[exer])

    all_set = []
    for index, row in dataset.iterrows():
        student = row['user_id']
        exer = row['problem_id']
        score = row['is_correct']
        all_set.append([int(student), int(exer),int(score)])

    bound = int(0.8*len(all_set))
    train_set = all_set[:bound]
    test_set = all_set[bound:]
    print(len(train_set))
    print(len(test_set))
    pdb.set_trace()
    # np.save("../data/coarse/item2knowledge.npy",item2knowledge)
    train_set = np.array(train_set)
    np.save("../data/coarse/train_set.npy", train_set)
    test_set = np.array(test_set)
    np.save("../data/coarse/test_set.npy", test_set)


def divide_data_ood():
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set and test_set (0.8:0.2)
    :return:
    '''
    all_data_json = "../data/coarse/student-problem-coarse.json"
    problem_json = "../data/coarse/problem.json"
    # data = pd.read_csv(all_data_json)
    with open(all_data_json, 'r', encoding='utf-8') as f_in:
        json_data = json.loads(f_in.read())
        # for line in f_in.readlines():
        #     dic = json.loads(line)
        #     json_data.append(dic)

    df_nested_list = pd.json_normalize(json_data, record_path=['seq'])
    raw_question = df_nested_list.problem_id.unique().tolist()
    num_skill = len(raw_question)
    # problem map: (start from 1)
    map_problems = {p: i + 1 for i, p in enumerate(raw_question)}
    num_questions = num_skill

    with open(problem_json, 'r', encoding='utf-8')as f_in:
        problem_data = []
        for line in f_in.readlines():
            dic = json.loads(line)
            problem_data.append(dic)

    df_problem_detail = pd.json_normalize(problem_data)
    df_problem_detail_sub = df_problem_detail[df_problem_detail.problem_id.isin(raw_question)]

    raw_concept = set()
    for c in df_problem_detail_sub.exercise_id.tolist():
        raw_concept.add(c)
    num_concept = len(raw_concept)
    # concept map: (start from 1)
    map_concepts = {c: i + 1 for i, c in enumerate(raw_concept)}

    item2knowledge = {}
    for index, row in tqdm.tqdm(df_problem_detail_sub.iterrows()):
        problem_id = row['problem_id']
        concept = row['exercise_id']
        item2knowledge[map_problems[problem_id]] = map_concepts[concept]

    # saved
    item_df = pd.DataFrame({'item_id': item2knowledge.keys(), 'knowledge_code': item2knowledge.values()})
    # item_df.to_csv(args.saved_item_dir, sep=',', index=False, header=True)

    # 4. build user map
    raw_users = df_nested_list.user_id.unique().tolist()
    num_users = len(raw_users)
    # users map: (start from 1)
    map_users = {u: i + 1 for i, u in enumerate(raw_users)}
    # args.num_questions = num_users

    # 5. build dataset
    dataset = df_nested_list[['user_id', 'problem_id', 'is_correct']]
    dataset['user_id'] = dataset['user_id'].map(map_users)
    dataset['problem_id'] = dataset['problem_id'].map(map_problems)

    dataset = dataset.sample(frac=1)

    all_set_dict = defaultdict(set)
    for index, row in dataset.iterrows():
        student = row['user_id']
        exer = row['problem_id']
        score = row['is_correct']
        all_set_dict[student].add(item2knowledge[exer])

    train_set_dict = defaultdict(list)
    test_set_dict = defaultdict(list)
    for student, row in all_set_dict.items():
        line = list(row)
        bound = int(0.8*len(line))
        train_set_dict[student] = line[:bound]
        test_set_dict[student] = line[bound:]

    train_set = []
    test_set = []
    for index, row in dataset.iterrows():
        student = row['user_id']
        exer = row['problem_id']
        score = row['is_correct']
        if item2knowledge[exer] in train_set_dict[student]:
            train_set.append([int(student), int(exer),int(score)])
        if item2knowledge[exer] in test_set_dict[student]:
            test_set.append([int(student), int(exer),int(score)])

    print(len(train_set))
    print(len(test_set))
    pdb.set_trace()
    np.save("../data/coarse/item2knowledge.npy",item2knowledge)
    train_set = np.array(train_set)
    np.save("../data/coarse/train_set_ood.npy", train_set)
    test_set = np.array(test_set)
    np.save("../data/coarse/test_set_ood.npy", test_set)


if __name__ == '__main__':
    divide_data()
