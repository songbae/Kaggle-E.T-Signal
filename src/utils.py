import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd


def make_aug(): ## 새로운 라벨 만들기 
  h = 273
  w = 256
  new = pd.read_csv('index_data.csv')
  cnt = 0
  new_img = np.zeros((h, w*6), dtype=float)
  for idx, k in enumerate(range(len(new))):
    img = np.load(new.path[k])
    i = idx % 3
    temp = np.zeros((h, w*6), dtype=float)
    for j in range(6):
      temp[:, w*j:w*(j+1)] = img[j]
    new_img[(h*i)//3:h*(i+1)//3, :] = temp[(h*i)//3:h*(i+1)//3, :]
    if i == 0 and idx != 0:
      np.save(f'./input/new_img/{cnt}.npy', new_img)
      cnt += 1

def seed_everything(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    # os.envrion["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Optimizer(optimizer, lr, parameter):
    if optimizer == 'adam':
        return optim.Adam(parameter, lr=lr)
    elif optimizer == 'adamw':
        return optim.AdamW(parameter, lr=lr)
    elif optimizer == 'sgd':
        return optim.SGD(parameter, lr=lr)

def Sched(cfg,optimizer):
  if cfg.sched_type=='cosine':
    return optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg.epochs, eta_min=1e-6)
  elif cfg.sched_type=='platue':
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5, patience=2)
  elif cfg.sched_type=='multi':
    return optim.lr_scheduler.MultiStepLR(optimizer,gamma=0.2,milestones=[3,6])
  elif cfg.sched_type=='re':
    return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.epochs//2, T_mult=1, eta_min=1e-6, last_epoch=-1)
