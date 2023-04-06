import os
import re
import sys
import argparse
import itertools
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", default="../../dataset/now")
args = parser.parse_args()

global df_user
global df_media
def category_split(category, df_meta_filter, df_log_filter):
    for cate in category:
        media_ids = list(df_meta_filter[df_meta_filter['channel_category'] == cate].index)
        df_c = df_log_filter.loc[media_ids]
        df_c_split = df_c[df_c['split'] == "test"]
        df_c_cold_log = df_c_split[df_c_split['wc_split'] == f'c_test']
        df_c_warm_log = df_c_split[df_c_split['wc_split'] == f'w_test']
        
        json_total = df_c_split[["media_id","id_no", "score"]].to_dict(orient='records')
        json_cold = df_c_cold_log[["media_id","id_no", "score"]].to_dict(orient='records')
        json_warm = df_c_warm_log[["media_id","id_no", "score"]].to_dict(orient='records')

        with open(os.path.join(f"../dataset/now/split/test/total_{cate}.json"), mode="w") as io:
            json.dump(json_total, io)

        with open(os.path.join(f"../dataset/now/split/test/cold_{cate}.json"), mode="w") as io:
            json.dump(json_cold, io)

        with open(os.path.join(f"../dataset/now/split/test/warm_{cate}.json"), mode="w") as io:
            json.dump(json_warm, io)

def data_split(df_log, splits):
    df_split = df_log[df_log['split'] == splits]
    df_cold_log = df_split[df_split['wc_split'] == f'c_{splits}']
    df_warm_log = df_split[df_split['wc_split'] == f'w_{splits}']
    
    json_total = df_split[["media_id","id_no","score"]].to_dict(orient='records')
    json_cold = df_cold_log[["media_id","id_no","score"]].to_dict(orient='records')
    json_warm = df_warm_log[["media_id","id_no","score"]].to_dict(orient='records')
    
    with open(os.path.join(f"../dataset/now/split/{splits}/total.json"), mode="w") as io:
        json.dump(json_total, io)
        
    with open(os.path.join(f"../dataset/now/split/{splits}/cold.json"), mode="w") as io:
        json.dump(json_cold, io)
        
    with open(os.path.join(f"../dataset/now/split/{splits}/warm.json"), mode="w") as io:
        json.dump(json_warm, io)
    
    item_split = {
        "all_item": list(set(df_split.index)),
        "cold_item": list(set(df_cold_log.index)),
        "warm_item": list(set(df_warm_log.index)),
        "all_user": list(set(df_split['id_no'])),
        "cold_user": list(set(df_cold_log['id_no'])),
        "warm_user": list(set(df_warm_log['id_no'])),
    }
    with open(os.path.join(f"../dataset/now/split/{splits}/track_user.json"), mode="w") as io:
        json.dump(item_split, io, indent=4)

def _filtering(_id):
    return os.path.exists(os.path.join(args.save_path, f"contents/vision/{_id}.pt")) & os.path.exists(os.path.join(args.save_path, f"contents/audio/{_id}.pt"))

def u_id_to_list(u_id):
    instance = df_gt.loc[u_id]
    return {u_id: list(instance['media_id'])}

def main_split(args):
    df_meta = pd.read_parquet(os.path.join(args.save_path, "balance/v_meta")).set_index("media_id")
    df_log = pd.read_parquet(os.path.join(args.save_path, "balance/iter_filtered_qoe")).set_index("media_id")
    drop_items = []
    for i in df_meta.index:
        if _filtering(i):
            pass
        else:
            drop_items.append(i)
    df_meta_filter = df_meta.drop(drop_items)
    df_log_filter = df_log.drop(drop_items)
    df_log_filter['media_id'] = df_log_filter.index
    category = ['ARTIS', 'DRAMA', 'ENTER', 'NEWS', 'SPORT']
    data_split(df_log=df_log_filter, splits="train")
    data_split(df_log=df_log_filter, splits="valid")
    data_split(df_log=df_log_filter, splits="test")
    category_split(category, df_meta_filter, df_log_filter)

def main(args):
    main_split(args)
        
        
if __name__ == "__main__":
    main(args)