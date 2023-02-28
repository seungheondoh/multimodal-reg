import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import math
from recommenders.evaluation.python_evaluation import (
    map_at_k, ndcg_at_k, precision_at_k, recall_at_k
)

def evaluation(df_gt, df_pred):
    k = 10
    cols = {
        'col_user': 'id_no',
        'col_item': 'media_id',
        'col_rating': 'score',
        'col_prediction': 'rating'
    }

    eval_map = map_at_k(df_gt, df_pred, k=k, **cols)
    eval_ndcg = ndcg_at_k(df_gt, df_pred, k=k, **cols)
    eval_precision = precision_at_k(df_gt, df_pred, k=k, **cols)
    eval_recall = recall_at_k(df_gt, df_pred, k=k, **cols)
    
    return {
        "ndcg": eval_ndcg,
        "recall": eval_recall
    }

def get_gt(args, fname = "total.json"):
    test_total = json.load(open(os.path.join(args.data_path, f"now/split/test/{fname}"), 'r'))
    df_gt = pd.DataFrame(test_total)
    target_user = list(set(df_gt['id_no']))
    target_item = list(set(df_gt['media_id']))
    user_to_idx = {i:idx for idx, i in enumerate(target_user)}
    item_to_idx = {i:idx for idx, i in enumerate(target_item)}
    df_gt['media_id'] = df_gt['media_id'].map(item_to_idx)
    df_gt['id_no'] = df_gt['id_no'].map(user_to_idx)
    return df_gt, target_item, target_user

def get_topk_rank(x_u, x_i, topk=10):
    x_u = torch.from_numpy(x_u)
    x_i = torch.from_numpy(x_i)
    x_u = torch.nn.functional.normalize(x_u, dim=-1)
    x_i = torch.nn.functional.normalize(x_i, dim=-1)
    score_matrix = x_u @ x_i.T
    u2i_score, u2i_rank = torch.topk(score_matrix, k=topk)
    u2i_rank = u2i_rank.numpy()
    u2i_score = u2i_score.numpy()
    return u2i_rank, u2i_score

def get_prediction(args, target_item, target_user, x_i):
    user_embs = torch.load(os.path.join(args.data_path, "now/balance/cf_u.pt"))
    x_u = np.array([user_embs[i] for i in target_user])
    u2i_rank, u2i_score = get_topk_rank(x_u, x_i)
    pair_info = []
    for idx, top_idx in enumerate(u2i_rank):
        top_score = u2i_score[idx]
        for item, score in zip(top_idx, top_score):
            pair_info.append([idx, item, score])
    df_pred = pd.DataFrame(pair_info, columns=["id_no", "media_id", "rating"])
    return df_pred