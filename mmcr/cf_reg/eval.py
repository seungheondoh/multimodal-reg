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
parser.add_argument("--cls_type", default="both", type=str)

parser.add_argument("--mlp_dim", default=768, type=int)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--depth", default=12, type=int)
parser.add_argument("--tid", default=0, type=int)

parser.add_argument("--output_dim", default=128, type=int)
parser.add_argument("--v_dim", default=640, type=int)
parser.add_argument("--a_dim", default=768, type=int)
parser.add_argument("--t_dim", default=640, type=int)
parser.add_argument("--dropout", default=0.1, type=float)



args = parser.parse_args()

    
def contents_eval(args, save_dir):
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
    print("load weight!")
    test_loader = get_dataloader(args=args, split="TEST")
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu) 

    model.eval()
    epoch_end_loss = []
    item_embs = {}
    for data_iter_step, batch in enumerate(test_loader):
        media_id, meta_id, a_embs, v_embs, t_embs, cf_embs = batch            
        if args.gpu is not None:
            meta_id = meta_id.cuda(args.gpu, non_blocking=True)
            v_embs = v_embs.cuda(args.gpu, non_blocking=True)
            a_embs = a_embs.cuda(args.gpu, non_blocking=True)
            t_embs = t_embs.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            i_embs = model.inference(v_embs, a_embs, t_embs, meta_id) # flatten batch
        i_embs = i_embs.detach().cpu().numpy()
        item_embs.update({i:j for i,j in  zip(media_id, i_embs)})
    torch.save(item_embs, os.path.join(save_dir, "item_embs.pt"))
    return item_embs


def main():
    save_dir = f"exp/{args.embs_type}/{args.fusion_type}_{args.head}_{args.depth}_{args.mlp_dim}/{args.cls_type}/{args.modality}_{args.tid}"
    results = {}
    if args.modality == "cf":
        item_embs = torch.load(os.path.join(args.data_path, "now/balance/cf_i.pt"))
    else:
        item_embs = contents_eval(args, save_dir)

    for fname in tqdm(os.listdir(os.path.join(args.data_path, "now/split/test"))):
        if fname != "track_user.json":
            df_gt, target_item, target_user = get_gt(args, fname)
            x_i = np.array([item_embs[i] for i in target_item])
            df_pred = get_prediction(args, target_item, target_user, x_i)
            results[fname.replace(".json","")] = evaluation(df_gt, df_pred)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "seed_results.json"), mode="w") as io:
        json.dump(results, io, indent=4)

if __name__ == '__main__':
    main()

    