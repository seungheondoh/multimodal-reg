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
from mmcr.utils.train_utils import Logger, AverageMeter, ProgressMeter, EarlyStopping, save_hparams
from sklearn import metrics

parser = argparse.ArgumentParser(description='NOW Training')
parser.add_argument('--data_path', type=str, default="../../dataset/")
parser.add_argument("--data_type", default="now", type=str)
parser.add_argument('--framework', type=str, default="contrastive") # or transcription
parser.add_argument('--arch', default='transformer')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N')
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

def main():
    save_dir = f"exp/{args.embs_type}/{args.fusion_type}_{args.head}_{args.depth}_{args.mlp_dim}/{args.cls_type}/{args.modality}_{args.tid}"
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
    train_loader = get_dataloader(args=args, split="TRAIN")
    val_loader = get_dataloader(args=args, split="VALID")
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu) 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # only update cls
    earlystopping_callback = EarlyStopping(tolerance=30)
    logger = Logger(save_dir)
    save_hparams(args, save_dir)

    best_val_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, optimizer, epoch, logger, args)
        val_loss = validate(val_loader, model, epoch, args)
        logger.log_val_loss(val_loss, epoch)
        if val_loss < best_val_loss:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, f'{save_dir}/best.pth')
            best_val_loss = val_loss

        earlystopping_callback(val_loss, best_val_loss)
        if earlystopping_callback.early_stop:
            print("We are at epoch:", epoch)
            break

def train(train_loader, model, optimizer, epoch, logger, args):
    train_losses = AverageMeter('Train Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[train_losses],prefix="Epoch: [{}]".format(epoch))
    iters_per_epoch = len(train_loader)
    model.train()
    for data_iter_step, batch in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, data_iter_step / iters_per_epoch + epoch, args)
        media_id, meta_id, a_embs, v_embs, t_embs, cf_embs = batch            
        if args.gpu is not None:
            meta_id = meta_id.cuda(args.gpu, non_blocking=True)
            v_embs = v_embs.cuda(args.gpu, non_blocking=True)
            a_embs = a_embs.cuda(args.gpu, non_blocking=True)
            t_embs = t_embs.cuda(args.gpu, non_blocking=True)
            y = cf_embs.cuda(args.gpu, non_blocking=True)
        # compute output
        loss = model(v_embs, a_embs, t_embs, y, meta_id)
        train_losses.step(loss.item(), y.size(0))
        logger.log_train_loss(loss, epoch * iters_per_epoch + data_iter_step)
        logger.log_learning_rate(lr, epoch * iters_per_epoch + data_iter_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if data_iter_step % args.print_freq == 0:
            progress.display(data_iter_step)

def validate(val_loader, model, epoch, args):
    losses_val = AverageMeter('Valid Loss', ':.4e')
    progress_val = ProgressMeter(len(val_loader),[losses_val],prefix="Epoch: [{}]".format(epoch))
    model.eval()
    epoch_end_loss = []
    for data_iter_step, batch in enumerate(val_loader):
        media_id, meta_id, a_embs, v_embs, t_embs, cf_embs = batch
        if args.gpu is not None:
            meta_id = meta_id.cuda(args.gpu, non_blocking=True)
            v_embs = v_embs.cuda(args.gpu, non_blocking=True)
            a_embs = a_embs.cuda(args.gpu, non_blocking=True)
            t_embs = t_embs.cuda(args.gpu, non_blocking=True)
            y = cf_embs.cuda(args.gpu, non_blocking=True)
        # compute output
        with torch.no_grad():
            loss = model(v_embs, a_embs, t_embs, y, meta_id) # flatten batch
        epoch_end_loss.append(loss.detach().cpu())
        losses_val.step(loss.item(), y.size(0))
        if data_iter_step % args.print_freq == 0:
            progress_val.display(data_iter_step)
    val_loss = torch.stack(epoch_end_loss).mean(0, False)
    return val_loss
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


if __name__ == '__main__':
    main()

    