import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pandas
from glob import glob
import pandas as pd


def prepare_date():
    data = {'path': [], 'label': []}
    main_path = './input/train/*/*/*.jpg'
    train_img_list = glob(main_path)
    category_df = pd.read_csv('./input/category.csv')
    class_dict = dict(category_df.values[:, ::-1])
    for sub_path in train_img_list:
        temp = sub_path
        data['path'].append(temp)
        label = class_dict[temp.split('\\')[-2]]
        data['label'].append(label)
    new_df = pd.DataFrame(data)
    new_df.to_csv('./input/new_train.csv', index=False)


class programmers_dataset(Dataset):
    def __init__(self, df, transform=None, normalization='simple', mode='train'):
        self.df = df
        self.label = self.df[:, 1]
        self.csv = self.df[:, 0]
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index: int):
        img_path = self.csv[index]
        img = np.load(img_path)
        alpha = np.random.rand(1)
        cnt = index % 1563
        label = self.label[index]
        if alpha > 0.8 and label != 1 and self.mode == 'train':
          aug_img = np.load(f'./input/new_img/{cnt}.npy')
          temp = np.zeros((273, 256*6), dtype=float)
          for i in range(6):
            temp[:, 256*i:256*(i+1)] = img[i]
          img = np.concatenate((temp[:, :256*3], aug_img[:, 256*3:]))
          label = 1
        elif alpha > 0.5 and label != 0 and self.mode == 'train':
          cnt = index % 15000
          aug_img = np.load(f'./input/new_train_img/{cnt}.npy')
          temp = np.zeros((273, 256*6), dtype=float)
          for i in range(6):
            temp[:, 256*i:256*(i+1)] = img[i]
          img = np.concatenate((temp[:, :256*3], aug_img[:, 256*3:]))
          label = 0

        else:

          temp = np.zeros((273, 256*6), dtype=float)
          for i in range(6):
            temp[:, 256*i:256*(i+1)] = img[i]
          img = temp
        if self.transform:
            img = self.transform(image=img)['image']
        sample = {'image': img, 'label': label}
        return sample

class base_dataset(Dataset):
    def __init__(self, df, transform=None, spec_tfms=None,normalization='simple', mode='train'):
        self.df = df
        self.label = self.df[:, 1]
        self.csv = self.df[:, 0]
        self.mode = mode
        self.transform = transform
        self.spec_tfms=spec_tfms

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index: int):
        img_path = self.csv[index]
        img = np.load(img_path)
        alpha =np.random.rand(1)
        if alpha>0.5 and self.mode=='train':
          img=self.spec_tfms(img)
        img=img.astype(np.float32)
        img=np.vstack(img).transpose((1,0))
        label=self.label[index]
        if self.transform:
            img = self.transform(image=img)['image']
        sample = {'image': img, 'label': label}
        return sample


class programmers_dataset_test(Dataset):
    def __init__(self, df, transform=None, normalization='simple', mode='train'):
        self.df = df
        self.csv = self.df[:, 0]
        self.mode = mode
        self.transform = transform

    def __len__(self):

        return len(self.csv)

    def __getitem__(self, index: int):
        img_path = self.csv[index]
        img = np.load(img_path).astype(np.float32)
        img=np.vstack(img).transpose((1,0))
        if self.transform:
            img = self.transform(image=img)['image']
        sample = {'image': img}
        return sample


if __name__ == '__main__':
    data = programmers_dataset()
