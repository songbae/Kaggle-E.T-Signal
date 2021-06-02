import os
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
from src.config import *
from src.losses import *
from src.utils import *
from src.dataset import *
from functools import partial
from src.models import *
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import timm
import copy
import wandb
from sklearn.metrics import f1_score,roc_auc_score

def main():
    # train
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--df_dir', default='./input/new_train_df.csv', type=str)
    parser.add_argument('--log_dir',default='./log/checkpoint')
    parser.add_argument('--k_fold', default=5, type=int)
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched_type', default='cosine', type=str)
    parser.add_argument('--cls_num',default=1049, type=int)

    # dataset
    parser.add_argument('--spilt', default='label', type=str)
    parser.add_argument('--postfix',default='v1' ,type=str, required=True)

    # model
    parser.add_argument('--model', default='efficientnet_b0', type=str)

    # criterion

    args = parser.parse_args()
    cfg = Config(args,mode='train')

    seed_everything(cfg.seed)
    train(cfg,args)


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def train(cfg,args):
    wandb.init()
    wandb.config.update(args)
    device = cfg.device
    criterion = nn.CrossEntropyLoss()
    # model = custom_mixernet().to(device)
    kfold = StratifiedKFold(n_splits=cfg.k_fold, shuffle=True, random_state=cfg.seed)
    data = pd.read_csv(cfg.df_dir)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(data,data.iloc[:,1])):
        best_roc = 0
        # model = custom_net(cfg).to(device)    
        model = timm.create_model(f'{cfg.model}',pretrained=True, in_chans=1,num_classes=2 ).to(device)
        print(model)
        wandb.watch(model)
        optimizer = Optimizer(cfg.optim, cfg.lr, model.parameters())
        scheduler = Sched(cfg,optimizer)
        train_data = base_dataset(data.values[train_idx], transform=cfg.trn_tfms,spec_tfms=cfg.spec_tfms, mode='train')
        val_data = base_dataset(data.values[valid_idx], transform=cfg.val_tfms, mode='valid')
        train_dl = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4,drop_last=True)
        valid_dl = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=4,drop_last=True)
        for epoch in range(cfg.epochs):
            for phase in ['train', 'valid']:
                running_loss, running_acc,prediction,targets=[],[],[],[]
                if phase == 'train':
                    model.train()
                    now_dl = train_dl
                else:
                    model.eval()
                    now_dl = valid_dl
                with torch.set_grad_enabled(phase == 'train'):
                    with tqdm(now_dl, total=len(now_dl), unit='batch') as now_bar:
                        for idx, sample in enumerate(now_bar):
                            optimizer.zero_grad()
                            images, labels = sample['image'].type(torch.FloatTensor).to(device), sample['label'].type(torch.LongTensor).to(device)
                            outputs = model(images)
                            loss = criterion(outputs,labels)
                            _, preds = torch.max(outputs, 1)
                            running_acc.append(torch.sum(preds==labels).detach().cpu().numpy())
                            prediction.extend(preds.detach().cpu().numpy())
                            targets.extend(labels.detach().cpu().numpy())
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            acc = np.mean(running_acc)/cfg.batch_size*100
                            now_bar.set_postfix(phase=phase,run_acc=acc)
                    if phase == 'valid':
                        roc = roc_auc_score(targets, prediction)
                        print(roc)
                        if best_roc<roc:
                            best_roc = roc
                            best_model = copy.deepcopy(model.state_dict())
                            model_path =  f'{cfg.model}_{fold}.pth'
                            torch.save(best_model, model_path)
                            print('best_model_saved')
                    # if cfg.sched_type=='cosine' and phase=='train':
                    if  phase=='train':
                      scheduler.step()
                    if phase=='valid':
                      wandb.log({
                        'valid_acc':acc,
                        'valid_ROC_AUC':roc_auc_score(targets,prediction)
                      })
                    if phase=='train':
                      wandb.log({
                        'lr':get_learning_rate(optimizer)[0],
                        'train_acc':acc
                      })

if __name__ == '__main__':
    main()
