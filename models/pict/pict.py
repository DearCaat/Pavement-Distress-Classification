import torch.nn as nn
import torch
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from .swin import *
from .kmeans import kmeans,kmeans_predict
import sys
from copy import deepcopy
sys.path.append("...")
from utils import ModelEmaV3

class PicT(nn.Module):
    def __init__(self, backbone=nn.Module,cluster=None,dim=768,**kwargs):
        super().__init__()
        self.cluster_model = cluster
        self.cluster_distance = kwargs['cluster_distance']
        self.nor_index = kwargs['nor_index']
        self.cluster_num = None
        self.cluster_flip_sel = kwargs['cluster_flip_sel']

        if cluster == kmeans:
            self.cluster_num = kwargs['num_cluster']
            self.persistent_center = kwargs['persistent_center']
            self.register_parameter('cluster_centers',nn.Parameter(torch.zeros(size=(self.cluster_num,dim)),requires_grad=False))

        self.thr = kwargs.pop('select_cluster_thr')
        num_classes = kwargs.pop('num_classes')
        self.instance_feature_extractor=backbone
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # the patch head is a binary classification, which aim to detect the distress in patch. 
        # 0 is normal, 1 is the distress.
        self.head_instance = nn.Linear(dim, kwargs['ins_num_classes'])  

        # self.head = nn.Sequential(
        #     nn.Linear(dim,num_classes) if num_classes > 0 else nn.Identity()
        # )
        self.head = self.instance_feature_extractor.head

        self.soft_max = nn.Softmax(-1)

        # self._init_weights(self.head)
        self._init_weights(self.head_instance)
    
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def __get_cluster_feat_f(self,clusters_feat,cluster_num):
        B = len(clusters_feat)
        feats_tmp = []
        clusters_num = [] if cluster_num == None else [cluster_num] * B
        for b in range(B):
            if cluster_num is not None:
                C = cluster_num
            else:
                C = len(clusters_feat[b])
                clusters_num.append(C)
            
            for i in range(C):
                if b==0 and i == 0:
                    feats_tmp = self.avgpool(clusters_feat[b][i].transpose(0,1)).unsqueeze(0)
                else:
                    feats_tmp = torch.cat((feats_tmp,self.avgpool(clusters_feat[b][i].transpose(0,1)).unsqueeze(0)))
        return feats_tmp,clusters_num

    def __get_cluster_feat_mask(self,inst_feat,cluster_num,clusters_mask):
        '''
        clusters_mask    C B D
        '''
        B,N,D = inst_feat.shape
        clusters_num = [] if cluster_num == None else [cluster_num] * B
        feats_tmp = []
        if cluster_num is not None:
            C = cluster_num
        else:
            C = len(clusters_mask)
        for b in range(B):
            c = 0
            for i in range(C):
                f_tmp = inst_feat[b][clusters_mask[i][b]]
                if len(f_tmp) != 0:
                    c +=1
                    if len(feats_tmp) == 0:
                        feats_tmp = self.avgpool(f_tmp.transpose(0,1)).unsqueeze(0)
                    else:
                        feats_tmp = torch.cat((feats_tmp,self.avgpool(f_tmp.transpose(0,1)).unsqueeze(0)))  
            if cluster_num is None:
                clusters_num.append(c)
        return feats_tmp,clusters_num
    
    def __slim_classifier(self,clusters_feat,thr = 0.8,cluster_num=None,clusters_mask=None):
        B = len(clusters_feat)
        D = clusters_feat[0][0].size(-1)
        if clusters_mask is not None:
            feats_tmp,clusters_num = self.__get_cluster_feat_mask(clusters_feat,cluster_num,clusters_mask)
        else:
            feats_tmp,clusters_num = self.__get_cluster_feat_f(clusters_feat,cluster_num)
            
        feats_tmp = feats_tmp.view(-1,D)
        feats_tmp = self.head(feats_tmp)
        scores = self.soft_max(feats_tmp)

        if self.nor_index < 0:
            scores,_ = torch.max(scores,dim=-1)
        else:
            scores = 1 - scores[:,self.nor_index]

        ## 230521: TODO: BUG
        # if cluster_num is not None:
        #     if scores.size(0) == B * cluster_num:
        #         scores = scores.view(B,cluster_num)
        # else:
        cluster_num = None
        # try:
        #     scores = scores.view(B,cluster_num)
        # except:
        #     cluster_num = None
        # if the cluster number is fixed in the batch, use the batch-level ops rather than the for loop
        if cluster_num is not None:
            mask_max = scores.clone()
            try:
                mask_max[:,:] = 0
            except:
                print(scores.size())
            max_clu_index = torch.argmax(scores,dim=1).view(B,1)
            mask_max = mask_max.scatter_(1,max_clu_index,1) == 1
            # if the highest distress confidential scores is lower than the threshold, we think it is the normal image
            # for this case, we will select the cluster which has the lowest confidential score
            if not self.training and self.nor_index >= 0 and self.cluster_flip_sel:
                mask_min = scores.clone() 
                mask_min[:,:] = 0
                min_clu_index = torch.argmin(scores,dim=1).view(B,1)
                mask_min = mask_min.scatter_(1,min_clu_index,1) == 1
                inverse_mask = scores[mask_max] < thr
                mask_max[inverse_mask] = mask_min[inverse_mask]
            feats_tmp = feats_tmp.view(B,cluster_num,-1)
            feats = feats_tmp[mask_max]
        else:
            j=0
            mask_max = []
            for b in range(B):
                max_clu_index = torch.argmax(scores[j:j+clusters_num[b]])
                if not self.training and scores[j+max_clu_index] < thr and self.nor_index >= 0 and self.cluster_flip_sel:
                    max_clu_index = torch.argmin(scores[j:j+clusters_num[b]])
                if b == 0:
                    feats = feats_tmp[j:j+clusters_num[b]][max_clu_index]
                else:
                    feats = torch.cat((feats,feats_tmp[j:j+clusters_num[b]][max_clu_index]))
                mask_max += [max_clu_index]
                j = j+clusters_num[b]

        return feats.view(B,-1),clusters_num,mask_max,scores
    
    def __sklearn_cluster(self,inst_feature):
        B,N,D = inst_feature.shape

        if self.cluster_model == kmeans:
            for b in range(B):
                if self.training:
                    clu_labels,self.get_parameter('cluster_centers').data = self.cluster_model(X=inst_feature[b],num_clusters=self.cluster_num,device=torch.cuda.current_device(),cluster_centers = self.get_parameter('cluster_centers').data if self.get_parameter('cluster_centers').data.any() != 0 and self.persistent_center else [],tqdm_flag=False,distance=self.cluster_distance)
                else:
                    if self.persistent_center:
                        clu_labels = kmeans_predict(X=inst_feature[b],device=torch.cuda.current_device(),cluster_centers = self.get_parameter('cluster_centers').data,tqdm_flag=False,distance=self.cluster_distance)
                    else:
                        clu_labels,_ = self.cluster_model(X=inst_feature[b],num_clusters=self.cluster_num,device=torch.cuda.current_device(),tqdm_flag=False,distance=self.cluster_distance)
                clu_labels = torch.unsqueeze(clu_labels,dim=0)
                if b == 0:
                    clusters_idcs = clu_labels.clone()
                else:
                    clusters_idcs = torch.cat((clusters_idcs,clu_labels))

        #cluster mask for generate cluster features
        for i in range(self.cluster_num):
            cluster_mask = clusters_idcs - i == 0
            if i == 0:
                clusters_mask = cluster_mask.clone().unsqueeze(0)
            else:
                clusters_mask = torch.cat((clusters_mask,cluster_mask.clone().unsqueeze(0)))
        return clusters_idcs,clusters_mask

    def forward(self,x):

        # step 1, get the instance feat by backbone Network
        inst_feature=self.instance_feature_extractor.forward_features(x) #B*N*D
        B,N,D = inst_feature.shape

        # step 2, cluster 
        if self.cluster_model == kmeans:
            # kmeans cluster 
            cluster_num = self.cluster_num
            # find cluster features
            clusters_idcs,clusters_mask = self.__sklearn_cluster(inst_feature)
            clusters_feat = inst_feature

        # there is not fixed cluster number in the test phase.
        if not self.training:
            cluster_num = None

        # step 3 classify
        # patch classify
        logits_inst = self.head_instance(inst_feature.view(-1,D))
        logits_inst = logits_inst.view(B,N,-1)

        # slim image classify
        if self.cluster_model is not None:
            logits_bag,clusters_num,_,_= self.__slim_classifier(clusters_feat,thr=self.thr,cluster_num = cluster_num,clusters_mask=clusters_mask)
        else:
            logits_bag,clusters_num = self.instance_feature_extractor.forward_head(inst_feature),[B]
            #logits_bag,clusters_num = self.head(avg_bag_feature),[B]
    
        return logits_bag, logits_inst,clusters_num

def __teacher_init(config,teacher):
    cpt = torch.load(config.PICT.TEACHER_INIT, map_location='cpu')
    std = cpt['state_dict']
    if config.PICT.INST_NUM_CLASS == config.MODEL.NUM_CLASSES:
        std_ins = dict([('weight',std['head.weight']),('bias',std['head.bias'])])
        teacher.head_instance.load_state_dict(std_ins, strict=True)
        teacher.head.load_state_dict(std_ins, strict=True)
    teacher.instance_feature_extractor.load_state_dict(std, strict=False)

@register_model
def pict(config,backbone=None,logger=None,**kwargs):
    if config.PICT.CLUSTER.NAME is None:
        cluster = None
    elif config.PICT.CLUSTER.NAME.lower() == 'kmeans':
        cluster = kmeans
    
    pict_para = {
        'num_cluster':config.PICT.CLUSTER.NUM_CLUSTER,
        'num_classes':config.MODEL.NUM_CLASSES,
        'ins_num_classes': config.PICT.INST_NUM_CLASS,
        'cluster_distance': config.PICT.CLUSTER.CLUSTER_DISTANCE.lower(),
        'cluster_thr': config.PICT.CLUSTER.THR,
        'select_cluster_thr': config.PICT.CLUSTER.SELECT_THR,
        'nor_index': config.PICT.CLUSTER.NOR_INDEX if hasattr(config.PICT.CLUSTER, 'NOR_INDEX') else config.DATA.CLS_NOR_INDEX,
        'persistent_center': config.PICT.CLUSTER.PERSISTENT_CENTER,
        'cluster_flip_sel': config.PICT.TEST_CLU_FLIP_SEL
    }

    student = PicT(backbone=backbone,cluster=cluster,**pict_para)
    teacher = deepcopy(student)
    
    if config.PICT.TEACHER_INIT:
        # __teacher_init(config,teacher)
        try:
            __teacher_init(config,teacher)
            logger.info(f"Teacher model inited")
        except Exception as e:
            logger.error(f"Teacher model init failed, Error: {e}")

    teacher = ModelEmaV3(teacher,decay=config.PICT.EMA_DECAY,device='cpu' if config.PICT.EMA_FORCE_CPU else None, diff_layers=[] if config.PICT.EMA_DIFF is None else config.PICT.EMA_DIFF,decay_diff=config.PICT.EMA_DECAY_DIFF)
    
    return {'main':student,'teacher':teacher}
