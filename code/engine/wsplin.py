import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
import math
import torch
import random as native_random

from timm.utils import *
from .iNet_cls import INetClsEngine

import sys
sys.path.append("..")
from utils import getDataByStick
class WSPLINEngine(INetClsEngine):
    def __init__(
        self,
        config,
        **kwargs):
        super().__init__(config)

        self.sm_index = None
        self.sm_test_index = None
        self.device = 'cuda:'+str(config.LOCAL_RANK) if torch.cuda.is_available() else 'cpu'
        self.sm_index = self.__sparse_mask(config)
        self._test_sparse_mask(config)
        
        self.pr_stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        self._binary_test = config.MODEL.NUM_CLASSES == 2 or config.TEST.BINARY_MODE 

        self.train_metrics = OrderedDict([
            ('sparse_loss',AverageMeter()),
            ('cls_loss',AverageMeter())
        ])
        self.train_metrics_epoch_log =['sparse_loss','cls_loss']
        self.train_metrics_iter_log =['sparse_loss','cls_loss']

        if self._binary_test:
        # 添加p@r评价指标
            self.test_metrics.update(OrderedDict([
                    ('p@r90',.0),
                    ('p@r95',.0),
                    ('p@r99',.0),
            ]))
            self.test_metrics_epoch_log += ['p@r90','p@r95','p@r99']
    def _test_sparse_mask(self,config):
        if self.sm_test_index is not None:
            return
        self.sm_test_index = []
        for i in range(config.WSPLIN.RANDSM_TEST_NUM):
            self.sm_test_index.append(self.__sparse_mask(config))
            
    def __sparse_mask(self,config):
        device = self.device
        if config.WSPLIN.RANDSM and config.DATA.IS_IP:
            _index = [native_random.sample(range(12),math.ceil(12 * config.WSPLIN.SPARSE_RATIO)),native_random.sample(range(12,16),math.ceil(4 * config.WSPLIN.SPARSE_RATIO)),[16]]
            _sm_index = torch.tensor([ind for st in _index for ind in st],device=device)
        else:
            if config.WSPLIN.SPARSE_RATIO != 1.:
                if config.WSPLIN.SPARSE_RATIO == .75:
                    _sm_index = torch.tensor([1,3,4,5,6,7,8,9,10,13,14,15,16],device=device)
                elif config.WSPLIN.SPARSE_RATIO == .5:
                    _sm_index = torch.tensor([1,3,5,7,9,10,13,14,16],device=device)
                elif config.WSPLIN.SPARSE_RATIO == .25:
                    _sm_index = torch.tensor([1,5,10,14,16],device=device)
            else:
                _sm_index = None
        return _sm_index
    def l1_regularizer(self,input_features,classes):
        if input_features.size(0)>0:
            l1_loss = torch.norm(input_features, p=1)
            input_features = input_features.view(input_features.size(0),-1,classes)
            #return l1_loss / (input_features.size(0) * input_features.size(1))
            return l1_loss / (input_features.size(0))
        else:
            return 0

    def cal_loss_func(self,config,models,idx,samples,targets,epoch,num_steps,criterions,**kwargs):
        # 处理WSPLIN-SS
        # randsm
        if self.sm_index is not None:
            samples = samples.index_select(1,self.sm_index)

        feature_maps = models['main'].feature_extract(samples)
        bs = samples.size(0)
        predictions = models['main'].classifier(feature_maps)

        if config.WSPLIN.SPARSE_LOSS_RANGE == 'normal':
            feature_maps = feature_maps[torch.where(targets == config.DATA.CLS_NOR_INDEX), True,False]
        elif config.WSPLIN.SPARSE_LOSS_RANGE == 'disease':
            feature_maps = feature_maps[torch.where(targets != config.DATA.CLS_NOR_INDEX), True,False]

        s_loss = float(config.WSPLIN.SPARSE_LOSS_ALPHA) * self.l1_regularizer(feature_maps,config.MODEL.NUM_CLASSES)

        cls_loss = criterions[0](predictions,targets)
        loss = cls_loss + s_loss
        metrics_values = OrderedDict([
            ('sparse_loss',[s_loss,len(targets)]),
            ('cls_loss',[cls_loss,targets.size(0)]),
        ])

        return loss,metrics_values,OrderedDict([])

    def update_per_epoch(self,config,epoch,**kwargs):
        self.sm_index = self.__sparse_mask(config)
        return

    def measure_per_iter(self,config,models,samples,targets,criterions,**kwargs):
        # 处理WSPLIN-SS
        # randsm
        output = None
        if config.WSPLIN.RANDSM:
            for i in range(config.WSPLIN.RANDSM_TEST_NUM):
                samples_masked = samples.index_select(1,self.sm_test_index[i])
                # compute output
                if output is None:
                    output = models['main'](samples_masked)
                else:
                    output += models['main'](samples_masked)

            output /= config.WSPLIN.RANDSM_TEST_NUM
        else:
            if self.sm_index is not None:
                samples = samples.index_select(1,self.sm_index)

            # compute output
            output = models['main'](samples)

        if isinstance(output, (tuple, list)):
            predition = output[0]
        else:
            predition = output

        output_soft = torch.nn.functional.softmax(predition,dim=-1)

        loss = criterions[0](predition, targets)
            
        pred = output_soft.cpu().numpy()
        label = targets.cpu().numpy()

        # topk acc cls
        acc1,acc5 = accuracy(output, targets, topk=self.topk)

        metrics_values = OrderedDict([
        ('acc1',[acc1,targets.size(0)]),
        ('acc5',[acc5,targets.size(0)]),
        ])

        others = OrderedDict([])

        return loss,pred,label,metrics_values,others
    def measure_per_epoch(self,config,**kwargs):
        assert not(config.MODEL.NUM_CLASSES == 2 and config.TEST.BINARY_MODE)

        metrics_values = OrderedDict([])
        
        
        label = kwargs['label']
        pred = kwargs['pred']

        ma_f1 = f1_score(label,np.argmax(pred,axis=1),average='macro')
        mi_f1 = f1_score(label,np.argmax(pred,axis=1),average='micro')

        if self._binary_test:
            if config.TEST.BINARY_MODE:
                label = label!=config.DATA.DATA_NOR_INDEX
                pred = 1-pred[:,config.DATA.DATA_NOR_INDEX]
            elif config.MODEL.NUM_CLASSES == 2:
                ma_f1 = f1_score(label,np.argmax(pred,axis=1),average='binary')
                mi_f1 = ma_f1
                pred = 1-pred[:,config.DATA.CLS_NOR_INDEX]

            precision,recall,thr=precision_recall_curve(label, pred)
            patr=getDataByStick([precision,recall],self.pr_stick)
            auc = roc_auc_score(label,pred)
            metrics_values.update(OrderedDict([
                ('auc',auc),
                ('p@r90',patr[2][0]),
                ('p@r95',patr[1][0]),
                ('p@r99',patr[0][0]),
            ]))

        metrics_values.update(OrderedDict([
        ('macro_f1',ma_f1),
        ('micro_f1',mi_f1)
        ]))
        others = OrderedDict([
        ])

        return metrics_values,others
