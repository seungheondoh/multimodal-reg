import os
import json
import math
import argparse
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

import torch
import torch.backends.cudnn as cudnn
# backbones
from mmcr.cf_reg.model import CFREG
from mmcr.cf_reg.loader.data_manger import get_dataloader
from mmcr.utils.eval_utils import get_gt, get_prediction, evaluation
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from mmcr.preprocessing.utils import get_models, download_from_vod_cloud, check_valid_video, get_nlp_embs
from mmcr.constants import (NOW_CHANNEL, CREATOR)

import random
random.seed(42)
torch.manual_seed(42)
cudnn.deterministic = True

parser = argparse.ArgumentParser(description='NOW Training')
parser.add_argument('--data_path', type=str, default="../../dataset/")
parser.add_argument("--data_type", default="now", type=str)
parser.add_argument('--framework', type=str, default="contrastive") # or transcription
parser.add_argument('--arch', default='transformer')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2048, type=int, metavar='N')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=1e-9, type=float)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=7, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=100, type=int)
# downstream options
parser.add_argument("--modality", default="vat", type=str)
parser.add_argument("--embs_type", default="average", type=str)
parser.add_argument("--fusion_type", default="transformer", type=str)
parser.add_argument("--cls_type", default="channel", type=str)

parser.add_argument("--mlp_dim", default=768, type=int)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--depth", default=12, type=int)
parser.add_argument("--tid", default=0, type=int)

parser.add_argument("--output_dim", default=128, type=int)
parser.add_argument("--v_dim", default=640, type=int)
parser.add_argument("--a_dim", default=768, type=int)
parser.add_argument("--t_dim", default=640, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--v_id", default="EB930D0AAD33EFA135FDF50A90352DF3E3D1", type=str)
args = parser.parse_args()

import re

def text_preprocessing(sentence):
    clean_text = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', sentence)
    list_of_token = [token.strip() for token in clean_text.split() if len(token.strip()) > 0]
    return " ".join(list_of_token)

def get_text_feature(instance):
    v_id = instance.name
    net_res = ", ".join([i['surface'] for i in instance['net_res_list']])
    title = instance['title']
    _texts = title + " " + net_res
    texts = text_preprocessing(_texts)
    with torch.no_grad():
        text_features = text_model.forward(texts, tokenizer)
    text_features = text_features.detach().cpu()
    return text_features

def video_to_wav(audio, a_processor, hop):
    audio_numpy = audio.to_soundarray(buffersize=16000).mean(axis=1)
    target_audio = (16000 * 10) * hop
    if audio_numpy.shape[0] > target_audio:
        audio_numpy = audio_numpy[:target_audio]
    else:
        pad = np.zeros(target_audio)
        pad[:audio_numpy.shape[0]] = audio_numpy
    audio_batch = np.split(audio_numpy, hop)
    audio_tensor = a_processor(audio_batch, sampling_rate=16000, return_tensors="pt")
    return audio_tensor['input_values']

def video_to_pixel(video, v_processor):
    video_batch = [v_processor(Image.fromarray(frame)) for frame in video.iter_frames()]
    video_tensor = torch.stack(video_batch)
    return video_tensor

def get_fusion_model(args, save_dir):
    model = CFREG(
        modality = args.modality,
        fusion_type = args.fusion_type,
        head = args.head,
        depth = args.depth, 
        v_dim = args.v_dim,
        a_dim = args.a_dim,
        t_dim = args.t_dim,
        mlp_dim = args.mlp_dim,
        output_dim = args.output_dim,
        dropout = args.dropout,
        loss_fn = torch.nn.MSELoss(),
        cls_type = args.cls_type
    )
    checkpoint= torch.load(os.path.join(save_dir,"best.pth"), map_location='cpu')
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu) 
    return model


def main():
    save_dir = f"exp/{args.embs_type}/{args.fusion_type}_{args.head}_{args.depth}_{args.mlp_dim}/{args.cls_type}/{args.modality}_{args.tid}"
    fl = pd.read_parquet(os.path.join(args.data_path, 'now',"balance","v_meta")).set_index("media_id")
    instance = fl.loc[args.v_id]
    audio, video, hop = download_from_vod_cloud(args.v_id, fps=1)
    audio_processor, audio_model, tokenizer, text_model, video_processor, video_model = get_models()
    fusion_model = get_fusion_model(args, save_dir)

    audio_model = audio_model.to(args.gpu)
    text_model = text_model.to(args.gpu)
    video_model = video_model.to(args.gpu)
    # get backbone embedding
    text_tensor, v_id = get_text_feature(instance)
    audio_tensor = video_to_wav(audio, audio_processor, hop)
    video_tensor = video_to_pixel(video, video_processor)
    # get token
    if args.cls_type == "channel":
        channel_meta = instance['channel_category']
        label_to_idx = {i:idx + 1 for idx, i in enumerate(NOW_CHANNEL)}
    elif args.cls_type == "creator":
        creator_meta = instance['channel_nm']
        label_to_idx = {i:idx + 1 for idx, i in enumerate(CREATOR)}
    elif args.cls_type == "both":
        channel_meta = instance['channel_category']
        creator_meta = instance['channel_nm']
        label_to_idx = {i:idx + 1 for idx, i in enumerate(NOW_CHANNEL + CREATOR)}
    else:
        print(f"{cls_type} type training")

    with torch.no_grad():
        text_embs = text_model.forward(text_tensor.to(args.gpu), tokenizer)
        audio_embs = audio_model(audio_tensor.to(args.gpu))
        video_embs = video_model.encode_image(video_tensor.to(args.gpu))
        print(text_embs.shape, audio_embs.shape, video_embs.shape)

        video_embs = video_embs.mean(0, False)
        last_hidden_states = audio_embs.last_hidden_state
        audio_embs = last_hidden_states.mean(1, False)
        print(text_embs.shape, audio_embs.shape, video_embs.shape)

        meta_id = torch.LongTensor(label_to_idx[channel_meta]).to(args.gpu)
        i_embs = model.inference(video_embs, audio_embs, text_embs, meta_id) # flatten batch
        i_embs = i_embs.detach().cpu().numpy()
    

if __name__ == '__main__':
    main()

    