from timm.models.efficientnet import efficientnet_b0
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import timm


class effnet(nn.Module):
    def __init__(self):
        super(effnet, self).__init__()
        self.enet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1,in_chans=1)

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.enet(x)
        return x


class effnetV2(nn.Module):
    def __init__(self):
        super(effnetV2, self).__init__()
        self.enet = timm.create_model(
            'nfnet_l0', pretrained=True, num_classes=2, in_chans=1)

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.enet(x)
        return x


class custom_net(nn.Module):
    '''
    'image_size', 'patch_size', 'dim', 'depth', and 'num_classes'

    '''
    def __init__(self,cfg):
        super(custom_net, self).__init__()
        self.model=cfg.model
        self.mixer = timm.create_model(self.model, in_chans=1, pretrained=True)
        self.in_features=self.mixer.classifier.in_features
        self.mixer.classifier=nn.Linear(in_features=self.in_features, out_features=2, bias=True)

    def forward(self, x):
        x = self.mixer(x)
        return x


class mlp_mixernet(nn.Module):
    '''
    'image_size', 'patch_size', 'dim', 'depth', and 'num_classes'

    '''
    def __init__(self):
        super(mlp_mixernet, self).__init__()
        self.mixer = timm.create_model('mixer_b16_224_in21k',pretrained=True)
        self.conv1 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.mixer.head = nn.Linear(in_features=768, out_features=2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mixer(x)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ArcMarginProduct(nn.Module):  # 어렵네 무슨 말이지?
    def __init__(self, in_feats, cls_num):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(cls_num, in_feats))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, logits):
        cosine = F.linear(F.normalize(logits), F.normalize(self.weight))
        return cosine


def gem(x, p=3, eps=1e-6):  # 얘는 멀까?
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super().__init__()

        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Backbone(nn.Module):
    def __init__(self, name='dm_nfnet_f3', pretrained=True):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained)

        if 'dm_nfnet' in name:
            self.out_feats = self.net.head.fc.in_features
        if 'efficientnet' in name:
            self.out_feats = self.net.classifier.in_features
        else:
            raise AttributeError('Only support [dm, eff ]')
        self.net.reset_classifier(0, '')

    def forward(self, x):
        x = self.net(x)
        return x


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.md = cfg.multi_dropout
        self.backbone = Backbone(
            name=cfg.backbone_name, pretrained=cfg.backbone_pretrained)

        if cfg.pool == 'gem':
            self. global_pool = GeM(p_trainable=cfg.p_trainable)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        if self.md:
            self.dropouts = nn.ModuleList(
                [nn.Dropout(cfg.multi_dropout_prob) for _ in range(cfg.multi_dropout_num)])
        if cfg.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_feats, cfg.embed_size, bias=True),
                nn.BatchNorm1d(cfg.embed_size),
                nn.PReLU()
            )
        elif cfg.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_feats, cfg.embed_size, bias=True),
                nn.BatchNorm1d(cfg.embed_size),
                nn.PReLU()
            )
        else:
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_feats, cfg.embed_size, bias=False),
                nn.BatchNorm1d(cfg.embed_size),
            )

        self.head = ArcMarginProduct(cfg.embed_size, cfg.cls_num)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        x = self.neck(x)

        if self.md:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(x.clone())
                    out = self.head(out)
                else:
                    temp_out = dropout(x.clone())
                    out += self.head(temp_out)
            return out/len(self.dropouts)
        else:
            return self.head(x)
