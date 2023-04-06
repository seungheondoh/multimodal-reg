import os
import bz2
import csv
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", default="../../../dataset/music4all-onion")
parser.add_argument("--save_path", default="../../../dataset/features")
parser.add_argument("--features", default="vision")
args = parser.parse_args()

def save_embs(save_path, feature_path, target_set):
    save, error = [], []
    with bz2.open(feature_path, mode='rt', newline='') as bzfp:
        ctr = 0
        for row in tqdm(csv.reader(bzfp)):
            item = row[0].split("\t")
            if ctr == 0:
                header = item
            else:
                fname = item[0]
                features = np.array(item[1:]).astype('float32')
                if fname in target_set:
                    save.append(fname)
                    np.save(os.path.join(save_path, fname + ".npy"), features)
                else:
                    error.append(fname)
            ctr += 1
    return save, error

import sys
import csv
csv.field_size_limit(sys.maxsize)

def main(args):
    item_meta = pd.read_csv("../../../dataset/music4all-cold/item_meta.csv", index_col=0)
    df_sort = item_meta.sort_values(by='min')
    target_set = set(df_sort.index)
    print(len(target_set))
    feature_dict = {
        "vision": "id_vgg19.tsv.bz2", 
        "audio": "id_ivec256.tsv.bz2", 
        "tags": "id_tags_tf-idf.tsv.bz2", 
        "lyrics": "id_lyrics_tf-idf.tsv.bz2",
        "genres": "id_genres_tf-idf.tsv.bz2"
        }
    feature_path = os.path.join(args.root_path, feature_dict[args.features])
    save_path = os.path.join(args.save_path, args.features)
    os.makedirs(save_path, exist_ok=True)
    save, error = save_embs(save_path, feature_path, target_set)
    print(len(save), len(error))
        
if __name__ == "__main__":
    main(args)