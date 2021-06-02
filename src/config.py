import os
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.transforms import VerticalFlip
from albumentations.core.composition import OneOf
import torch
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from audiomentations import SpecCompose,SpecChannelShuffle,SpecFrequencyMask

class Config:
    def __init__(self, args, mode='train'):
        self.main_dir = './'
        self.df_dir = args.df_dir
        # a_dir=os.path.join(self.main_dir,args.input_dir)
        self.k_fold = args.k_fold
        self.epochs = args.epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.postfix = args.postfix
        self.log_dir = args.log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        # criterion
        self.optim = args.optim
        self.lr = args.lr
        # scheduler
        self.sched_type = args.sched_type
        # model
        self.cls_num = args.cls_num
        self.model = args.model
        # self.checkpoint=self.main_dir/'checkpoints'/str(self.backbone_name+'_'+args.postfix)
        # if not os.path.exists(self.checkpoint):
        #   os.makedirs(self.checkpoint, exist_ok=True)

        self.mean = [0.56019358, 0.52410121, 0.501457]
        self.std = [0.23318603, 0.24300033, 0.24567522]
        self.spec_tfms=SpecCompose([
            SpecChannelShuffle(p=.7),
            SpecFrequencyMask(p=.2),
        ], p=1.0)
        self.trn_tfms = A.Compose([
            # A.Resize(224, 224, p=1.0),
            A.Resize(256,512),
            A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0)
            ],p=0.5),
            # SpecChannelShuffle(),
            # SpecFrequencyMask()
            ToTensorV2()
        ])
        self.val_tfms = A.Compose([
            A.Resize(256, 512),
            # A.Resize(256, 256, p=1.0),
            ToTensorV2()
        ])
        self.test_tfms = A.Compose([  # Test time augmentation
            A.Resize(256, 512),
            A.HorizontalFlip(p=0.7),
            A.GaussNoise(p=0.5),

            A.OneOf([
                    A.CLAHE(p=0.5),
                    A.Compose([
                        A.RandomBrightness(limit=0.5, p=0.6),
                        A.RandomContrast(limit=0.4, p=0.6),
                        A.RandomGamma(p=0.6),
                    ])
                    ], p=0.65),

            A.OneOf([
                    A.HueSaturationValue(10, 20, 10, p=1.0),
                    A.RGBShift(p=1.0),
                    A.Emboss(p=1.0),
                    ], p=0.5),

            # A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.3, p=0.3),

            A.OneOf([
                    A.Perspective(p=1.0, scale=(0.05, 0.1)),
                    A.GridDistortion(p=1.0, distort_limit=0.25, border_mode=0),
                    A.OpticalDistortion(p=1.0, shift_limit=0.1,
                                        distort_limit=0.1, border_mode=0)
                    ], p=0.65),

            A.Normalize(p=1.0, mean=self.mean, std=self.std),
            ToTensorV2()
        ])
