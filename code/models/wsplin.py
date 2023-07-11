import math
import os
import torch
import torch.nn as nn

from timm.models.registry import register_model
from timm.models.layers import  trunc_normal_

class ClassifierNetwork(nn.Module):
    def __init__(self, num_classes,patches,dp_rate=0.5):
        super().__init__()

        self.cls_head = nn.Sequential(
            nn.Linear(num_classes*patches, num_classes*patches),
            nn.ReLU(),
            nn.Dropout(p=dp_rate),
            nn.Linear(num_classes*patches, num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        return self.cls_head(x)


class WSPLIN_IP(nn.Module):
    def __init__(self,backbone,num_classes=8,patches=17,dp_rate=0.5,only_backbone=False):
        super().__init__()
        self.only_backbone = only_backbone
        self.backbone = backbone
        self.classifier = nn.Identity() if only_backbone else ClassifierNetwork(num_classes,patches,dp_rate)

    def feature_extract(self,x):
        bs = x.size(0)
        x = x.flatten(0,1)
        x = self.backbone(x)
        x = x.view(bs, -1)
        return x

    def forward(self, x):
        if not self.only_backbone:
            x = self.feature_extract(x)
            x = self.classifier(x)
        else:
            x = self.backbone(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.backbone, 'no_weight_decay'):
            return {'backbone.' + i for i in self.backbone.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.backbone, 'no_weight_decay_keywords'):
            return {'backbone.' + i for i in self.backbone.no_weight_decay_keywords()}
        return {}

@register_model
def wsplin(pretrained=False,backbone=None,**kwargs):
    config = kwargs.pop('config')
    backbone = backbone
    if config.MODEL.BACKBONE_INIT is not None:
        try:
            cpt = torch.load(config.MODEL.BACKBONE_INIT, map_location = 'cpu')
        except:
            cpt = torch.load(os.path.join(config.OUTPUT,'model',config.MODEL.BACKBONE_INIT), map_location = 'cpu')
            
        res = backbone.load_state_dict(cpt['state_dict'],strict=False)
        # print(res)
    patches_num = math.ceil(config.DATA.NUM_PATCHES * config.WSPLIN.SPARSE_RATIO)
    patches_num = patches_num if patches_num > 3 else 3
    return WSPLIN_IP(backbone,num_classes=config.MODEL.NUM_CLASSES,patches=patches_num,dp_rate=config.WSPLIN.CLS_HEAD_DP_RATE,only_backbone=config.WSPLIN.ONLY_BACKBONE)
