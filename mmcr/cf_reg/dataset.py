import os
import json
import random
import pickle
import torch
import numpy as np
from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset, DataLoader

class M4A_Dataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.cf_item = torch.load(os.path.join(self.data_path, "features/cf_i.pt"))
        self.get_file_list()

    def get_file_list(self):
        # ['all_item', 'train_warm_item', 'valid_warm_item', 'test_warm_item', 'valid_cold_item', 'test_cold_item']
        track_split = json.load(open(os.path.join(self.data_path, "music4all-cold", "sub_cold_split.json"), "r"))
        if self.split == "TRAIN":
            self.fl = track_split['train_track']
        elif self.split == "VALID":
            self.fl = track_split['valid_track']
        elif self.split == "TEST":
            self.fl = track_split['test_track']
        else:
            raise ValueError(f"Unexpected split name: {self.split}")
        del track_split
    
    def __getitem__(self, index):
        item = self.fl[index]
        audio = np.load(os.path.join(self.data_path, f"features/audio/{item}.npy"))
        vision = np.load(os.path.join(self.data_path, f"features/vision/{item}.npy"))
        lyrics = np.load(os.path.join(self.data_path, f"features/lyrics/{item}.npy"))
        genres = np.load(os.path.join(self.data_path, f"features/genres/{item}.npy"))
        # tags = np.load(os.path.join(self.data_path, f"features/tags/{item}.npy"))
        cf_vec = self.cf_item[item]
        return item,audio,vision,lyrics,genres,cf_vec

    def __len__(self):
        return len(self.fl)