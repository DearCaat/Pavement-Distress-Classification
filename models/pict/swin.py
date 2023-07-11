import logging

from timm.models.swin_transformer import SwinTransformer,checkpoint_filter_fn
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinforPict(SwinTransformer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forwad_head(x)
        return x

def _create_swin_transformer(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        SwinforPict, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

@register_model
def swin_pict_small_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return _create_swin_transformer('swin_small_patch4_window7_224', pretrained=pretrained, **model_kwargs)