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

    all_set = []
    for index, row in dataset.iterrows():
        student = row['user_id']
        exer = row['problem_id']
        score = row['is_correct']
        all_set.append([int(student), int(exer),int(score)])

    np.save("../data/coarse/item2knowledge.npy",item2knowledge)

    bound = int(0.2*len(all_set))
    random.shuffle(all_set)
    set1,set2,set3,set4,set5 = all_set[:bound],all_set[bound:bound*2],all_set[bound*2:bound*3],all_set[bound*3:bound*4],all_set[bound*4:]

    tmp = []
    tmp.extend(set1)
    tmp.extend(set2)
    tmp.extend(set3)
    tmp.extend(set4)
    cut = int(len(tmp)/8*7)
    train_set = tmp[:cut]
    eval_set = tmp[cut:]
    test_set = set5
    train_set = np.array(train_set)
    np.save("../data/coarse/train_set1.npy", train_set)
    test_set = np.array(test_set)
    np.save("../data/coarse/test_set1.npy", test_set)
    eval_set = np.array(eval_set)
    np.save("../data/coarse/eval_set1.npy", eval_set)

    tmp = []
    tmp.extend(set1)
    tmp.extend(set2)
    tmp.extend(set3)
    tmp.extend(set5)
    cut = int(len(tmp)/8*7)
    train_set = tmp[:cut]
    eval_set = tmp[cut:]
    test_set = set4
    train_set = np.array(train_set)
    np.save("../data/coarse/train_set2.npy", train_set)
    test_set = np.array(test_set)
    np.save("../data/coarse/test_set2.npy", test_set)
    eval_set = np.array(eval_set)
    np.save("../data/coarse/eval_set2.npy", eval_set)


    tmp = []
    tmp.extend(set1)
    tmp.extend(set2)
    tmp.extend(set4)
    tmp.extend(set5)
    cut = int(len(tmp)/8*7)
    train_set = tmp[:cut]
    eval_set = tmp[cut:]
    test_set = set3
    train_set = np.array(train_set)
    np.save("../data/coarse/train_set3.npy", train_set)
    test_set = np.array(test_set)
    np.save("../data/coarse/test_set3.npy", test_set)
    eval_set = np.array(eval_set)
    np.save("../data/coarse/eval_set3.npy", eval_set)

    tmp = []
    tmp.extend(set1)
    tmp.extend(set3)
    tmp.extend(set4)
    tmp.extend(set5)
    cut = int(len(tmp)/8*7)
    train_set = tmp[:cut]
    eval_set = tmp[cut:]
    test_set = set2
    train_set = np.array(train_set)
    np.save("../data/coarse/train_set4.npy", train_set)
    test_set = np.array(test_set)
    np.save("../data/coarse/test_set4.npy", test_set)
    eval_set = np.array(eval_set)
    np.save("../data/coarse/eval_set4.npy", eval_set)


    tmp = []
    tmp.extend(set2)
    tmp.extend(set3)
    tmp.extend(set4)
    tmp.extend(set5)
    cut = int(len(tmp)/8*7)
    train_set = tmp[:cut]
    eval_set = tmp[cut:]
    test_set = set1
    train_set = np.array(train_set)
    np.save("../data/coarse/train_set5.npy", train_set)
    test_set = np.array(test_set)
    np.save("../data/coarse/test_set5.npy", test_set)
    eval_set = np.array(eval_set)
    np.save("../data/coarse/eval_set5.npy", eval_set)

if __name__ == '__main__':
    divide_data()
