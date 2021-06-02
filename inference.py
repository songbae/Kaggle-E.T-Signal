from numpy.core.defchararray import index
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader,Dataset
import pandas as pd 
import numpy as np 
from src.models import *
from src.dataset import * 
from src.utils import *
from src.config import *
import argparse
from tqdm import tqdm
def test():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--df_dir', default='./input/new_test.csv', type=str)
    parser.add_argument('--log_dir', default='./log/checkpoint')
    parser.add_argument('--k_fold', default=4, type=int)
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched_type', default='cosine', type=str)
    parser.add_argument('--decay', default=0, type=float)
    parser.add_argument('--cls_num', default=2, type=int)

    # dataset
    parser.add_argument('--spilt', default='label', type=str)
    parser.add_argument('--postfix', default='efficientnetv2_rw_s', type=str)

    # model
    parser.add_argument('--model', default='efficientnetv2_rw_s', type=str)

    # criterion

    args = parser.parse_args()
    cfg = Config(args, mode='test')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    import pandas as pd
    model_n = ['efficientnetv2_rw_s_0.pth','efficientnetv2_rw_s_1.pth', 'efficientnetv2_rw_s_2.pth', 'efficientnetv2_rw_s_3.pth']
    k_fold = 4
    models=list()
    for i in range(k_fold):
      model = timm.create_model( f'{cfg.model}', pretrained=True, in_chans=1, num_classes=2).to(device)
      model.load_state_dict(torch.load(model_n[i]))
      models.append(model.to(device))
    submission = pd.read_csv('./input/sample_submission.csv')
    test_data=pd.read_csv('./input/new_test_df.csv')
    test_dataset=programmers_dataset_test(test_data.values,transform=cfg.val_tfms,mode='test')
    test_dl=DataLoader(test_dataset,shuffle=False,batch_size=1,num_workers=4)
    labels=[]
    for idx, sample in tqdm(enumerate(test_dl)):
        inputs=sample['image'].type(torch.FloatTensor).to(device)
        outputs=0.0
        for model in models:
          model.eval()
          output=model(inputs)
          outputs+=output
        label = torch.argmax(outputs, axis=1).detach().cpu().numpy()
        labels.extend(label)
    submission.iloc[:,1]=labels
    submission.to_csv('augv1_sub.csv',index=False)
if __name__=='__main__':
  test()
