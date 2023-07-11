import torch
import torch.nn as nn

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

import os

class IOPLIN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward_features(self,x):
        bs,ps = x.size(0),x.size(1)
        x = torch.flatten(x,0,1)
        x = self.backbone.forward_features(x)
        # unflatten = nn.Unflatten(0,(bs,ps))
        return x

    def forward(self, x):
        bs,ps = x.size(0),x.size(1)
        x = torch.flatten(x,0,1)
        # unflatten = nn.Unflatten(0,(bs,ps))
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

@register_model
def ioplin(pretrained=False,backbone=None,**kwargs):
    config = kwargs.pop('config')
    backbone = backbone
    if config.MODEL.BACKBONE_INIT is not None:
        try:
            cpt = torch.load(config.MODEL.BACKBONE_INIT, map_location = 'cpu')
        except:
            cpt = torch.load(os.path.join(config.OUTPUT,'model',config.MODEL.BACKBONE_INIT), map_location = 'cpu')
            
        res = backbone.load_state_dict(cpt['state_dict'],strict=False)
    return IOPLIN(backbone)