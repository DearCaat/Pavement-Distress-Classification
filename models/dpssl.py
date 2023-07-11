import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from timm.models.registry import register_model
from timm.models.layers import  trunc_normal_

class FeatureExtractorNetwork(nn.Module):
    def __init__(self,backbone):
        super(FeatureExtractorNetwork, self).__init__()

        self.model = backbone

    def forward(self, x):
        # input shape is [batch_size, patches, 3, 300, 300]
        bs = x.size(0)
        x = x.flatten(0,1)
        # x = x.view(-1, 3, 300, 300)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        x = self.model.forward_features(x)
        x = self._avg_pooling(x)
        return x


class SPFSN(nn.Module):
    def __init__(self,classes,dim_input=1536,dim_inner=384):
        super(SPFSN, self).__init__()
        self.classes = classes
        self.spfsn = nn.Sequential(
            nn.Linear(dim_input, dim_inner),
            nn.GELU(),
            nn.Linear(dim_inner, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.classes),
        )

        self.apply(_init_weights)
        

    def forward(self, x, bs):     
        H = x.view(bs,-1,self.L) # B*NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = A@H  # KxL
        M = M.squeeze()
        Y_prob = self.classifier(M)
        
        return Y_prob.view(bs,-1)

def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DPSSL(nn.Module):
    def __init__(self,backbone,num_classes=8,spfsn_dim_input=1536,spfsn_dim_inner=384):
        super().__init__()
        self.backbone = backbone
        self.feature_extractor = FeatureExtractorNetwork(backbone)
        self.classifier = SPFSN(num_classes,dim_input=spfsn_dim_input,dim_inner=spfsn_dim_inner)

    def forward(self, x,bs=0):
        x = self.feature_extractor(x)
        x = self.classifier(x,bs)
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
def dpssl_effi_b3(pretrained=False,**kwargs):
    config = kwargs.pop('config')
    if config.MODEL.BACKBONE_INIT is not None:
        pretrained = False
    img_size = kwargs.pop('img_size')
    backbone = create_model(
        model_name='tf_efficientnet_b3',
        pretrained=pretrained,
        **kwargs
    )
    if config.MODEL.BACKBONE_INIT is not None:
        try:
            cpt = torch.load(config.MODEL.MODELBACKBONE_INIT, map_location = 'cpu')
        except:
            cpt = torch.load(os.path.join(config.OUTPUT,'model',config.MODEL.BACKBONE_INIT), map_location = 'cpu')
            
        res = backbone.load_state_dict(cpt['state_dict'],strict=False)
        # print(res)
    return DPSSL(backbone,num_classes=config.MODEL.NUM_CLASSES,spfsn_dim_input=config.DPSSL.DIM_INPUT,spfsn_dim_inner=config.DPSSL.DIM_INNNER)

