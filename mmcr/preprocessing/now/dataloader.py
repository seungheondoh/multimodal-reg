import os
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import get_models, download_from_vod_cloud, check_valid_video, get_nlp_embs
from moviepy.editor import VideoFileClip
from PIL import Image


class Vision_Dataset(Dataset):
    def __init__(self, data_path, v_processor, target=None):
        self.data_path = data_path
        self.v_processor = v_processor
        self.black_list = [i.replace(".npy", "") for i in os.listdir(os.path.join(data_path, "contents", "error"))]
        self.fl = pd.read_parquet(os.path.join(data_path, "balance","v_meta")).set_index("media_id")

    def __getitem__(self, index):
        instance = self.fl.iloc[index]
        v_id = instance.name
        try:
            _, video, _ = download_from_vod_cloud(v_id)
            video_tensor = self.video_to_pixel(video)
            return video_tensor, v_id
        except:
            np.save(f"./remove/{v_id}.npy", v_id)
            return "error", "error"

    def video_to_pixel(self, video):
        video_batch = [self.v_processor(Image.fromarray(frame)) for frame in video.iter_frames()]
        video_tensor = torch.stack(video_batch)
        return video_tensor

    def __len__(self):
        return len(self.fl)

class Audio_Dataset(Dataset):
    def __init__(self, data_path, a_processor, target=None):
        self.data_path = data_path
        self.a_processor = a_processor
        self.black_list = [i.replace(".npy", "") for i in os.listdir(os.path.join(data_path, "contents", "error"))]
        self.fl = pd.read_parquet(os.path.join(data_path, "balance","v_meta")).set_index("media_id")

    def __getitem__(self, index):
        instance = self.fl.iloc[index]
        v_id = instance.name
        try:
            audio, video, hop = download_from_vod_cloud(v_id)
            audio_tensor = self.video_to_wav(audio, hop)
            return audio_tensor, v_id
        except:
            np.save(f"./remove/{v_id}.npy", v_id)
            return "error", "error"

    def video_to_wav(self, audio, hop):
        audio_numpy = audio.to_soundarray(buffersize=16000).mean(axis=1)
        target_audio = (16000 * 10) * hop
        if audio_numpy.shape[0] > target_audio:
            audio_numpy = audio_numpy[:target_audio]
        else:
            pad = np.zeros(target_audio)
            pad[:audio_numpy.shape[0]] = audio_numpy
        audio_batch = np.split(audio_numpy, hop)
        audio_tensor = self.a_processor(audio_batch, sampling_rate=16000, return_tensors="pt")
        return audio_tensor['input_values']

    def __len__(self):
        return len(self.fl)


class NLP_Dataset(Dataset):
    def __init__(self, data_path, target=None):
        self.data_path = data_path
        self.black_list = [i.replace(".npy", "") for i in os.listdir(os.path.join(data_path, "contents", "error"))]
        self.fl = pd.read_parquet(os.path.join(data_path, "balance","v_meta")).set_index("media_id")

    def __getitem__(self, index):
        instance = self.fl.iloc[index]
        v_id = instance.name
        net_res = ", ".join([i['surface'] for i in instance['net_res_list']])
        title = instance['title']
        _texts = title + " " + net_res
        texts = self.text_preprocessing(_texts)
        return texts, v_id

    def text_preprocessing(self, sentence):
        clean_text = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', sentence)
        list_of_token = [token.strip() for token in clean_text.split() if len(token.strip()) > 0]
        return " ".join(list_of_token)

    def __len__(self):
        return len(self.fl)