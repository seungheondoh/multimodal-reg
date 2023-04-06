import os
import re
import sys
import argparse
import itertools
import pandas as pd
import numpy as np
import torch


from tqdm import tqdm
import multiprocessing as mp
from utils import get_models, download_from_vod_cloud, check_valid_video, get_nlp_embs
from torch.utils.data import Dataset, DataLoader
from dataloader import Audio_Dataset, Vision_Dataset, NLP_Dataset

def video_extractor(args, video_model, video_processor, target):
    video_model = video_model.to(args.v_device)
    video_model.eval()
    dataset = Vision_Dataset(data_path=args.save_path, v_processor=video_processor, target=target)
    infer_dataloader = DataLoader(dataset, batch_size=1, shuffle=None, num_workers=4, pin_memory=True, drop_last=False)
    for batch in tqdm(infer_dataloader):
        video_tensor, v_id = batch
        if v_id[0] != "error":
            media_id = v_id[0]
            with torch.no_grad():
                image_features = video_model.encode_image(video_tensor.squeeze(0).to(args.v_device))
            image_embs_per_sec = image_features.detach().cpu()
            image_embs_per_10sec = torch.stack([i.mean(0, False) for i in image_embs_per_sec.split(10, dim=0)])
            torch.save(image_embs_per_10sec, os.path.join(args.save_path, "contents", "vision",f"{media_id}.pt"))


def audio_extractor(args, audio_model, audio_processor, target):
    audio_model = audio_model.to(args.a_device)
    audio_model.eval()
    dataset = Audio_Dataset(data_path=args.save_path, a_processor=audio_processor, target=target)
    infer_dataloader = DataLoader(dataset, batch_size=1, shuffle=None, num_workers=4, pin_memory=True, drop_last=False)
    for batch in tqdm(infer_dataloader):
        audio_tensor, v_id = batch
        if v_id[0] != "error":
            media_id = v_id[0]
            with torch.no_grad():
                audio_features = audio_model(audio_tensor.squeeze(0).to(args.a_device))
            last_hidden_states = audio_features.last_hidden_state
            audio_embs_per_10sec = last_hidden_states.detach().cpu().mean(1, False)
            torch.save(audio_embs_per_10sec, os.path.join(args.save_path, "contents", "audio", f"{media_id}.pt"))

def nlp_extractor(args, text_model, tokenizer, target):
    text_model.eval()
    dataset = NLP_Dataset(data_path=args.save_path, target=target)
    infer_dataloader = DataLoader(dataset, batch_size=1, shuffle=None, num_workers=4, pin_memory=True, drop_last=False)
    for batch in tqdm(infer_dataloader):
        texts, v_id = batch
        media_id = v_id[0]
        with torch.no_grad():
            text_features = text_model.forward(texts[0], tokenizer)
        text_features = text_features.detach().cpu()
        torch.save(text_features, os.path.join(args.save_path, "contents", "nlp", f"{media_id}.pt"))
    

def main(args):
    os.makedirs(os.path.join(args.save_path, "contents", "audio"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "contents", "vision"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "contents", "nlp"), exist_ok=True)
    audio_processor, audio_model, tokenizer, text_model, video_processor, video_model = get_models()
    target = list(pd.read_parquet(os.path.join(args.save_path, "balance", "v_meta"))['media_id'])
    not_target = [i.replace(".npy","")for i in os.listdir("./remove")]
    if args.modality == "audio":
        audio_extractor(args, audio_model, audio_processor,target=None)
    elif args.modality == "vision":
        video_extractor(args, video_model, video_processor,target=None)
    elif args.modality == "nlp":
        nlp_extractor(args, text_model, tokenizer, target=None)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="../../dataset")
    parser.add_argument("--modality", default="vision")
    parser.add_argument("--a_device", default="cuda:4")
    parser.add_argument("--v_device", default="cuda:5")
    args = parser.parse_args()
    main(args)