import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import *

from .wsplin import WSPLINEngine

class KDLoss(nn.Module):
    def __init__(self,T):
        super(KDLoss,self).__init__()
        self.T = T

    def forward(self,y_s,y_t):
        p_s = F.log_softmax(y_s/self.T,dim=-1)
        p_t = F.softmax(y_t/self.T,dim=-1)
        loss = F.kl_div(p_s,p_t,reduction='batchmean')
        # 其它人的版本，除了temp**2
        # loss = F.kl_div(p_s,p_t,reduction='sum') * (self.T**2) / y_s.size(0) 
        return loss

class DPSSLEngine(WSPLINEngine):
    def __init__(self,config,**kwargs):

        # 除了主损失以外的metric，主损失会每个iter进行log
        # 每个iter更新的指标需要初始化成AverageMeter
        # KD
        self.train_metrics = OrderedDict([
        ('loss_kd',AverageMeter()),
        ('loss_cls',AverageMeter()),
        ])
        self.train_metrics_epoch_log =['loss_kd','loss_cls']
        self.train_metrics_iter_log =['loss_kd','loss_cls']

        self.test_metrics = OrderedDict([
        ('acc1',AverageMeter()),
        ('acc5',AverageMeter()),
        ('macro_f1',.0),
        ('micro_f1',.0),
        ('auc',.0),])

        self.test_metrics_epoch_log =['macro_f1','micro_f1','auc']
        self.test_metrics_iter_log =['acc1','acc5']

        if config.TEST.BINARY_MODE:
            # 添加p@r评价指标
            self.test_metrics.update(OrderedDict([
                    ('p@r90',.0),
                    ('p@r95',.0),
                ]))

    def update_per_epoch(self,config,epoch,**kwargs):
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

    def cal_loss_func(self,config,models,idx,samples,targets,epoch,num_steps,criterions,**kwargs):
        predictions,loss_mim = models['main'](samples)
        loss_cls = criterions[0](predictions,targets)
        
        kd_criterion = KDLoss(config.DPSSL.KD_TEMP)
        if not config.DPSSL.KD_CPU:
            kd_criterion.cuda()
        with torch.no_grad():
            kd_logits = models['kd'](samples,bs=samples.size(0))
            kd_loss = kd_criterion(predictions,kd_logits)
            
        loss = loss_cls + config.DPSSL.KD_ALPHA * kd_loss

        metrics_values = OrderedDict([
            ('loss_mim',[loss_mim,targets.size(0)]),
            ('loss_cls',[loss_cls,targets.size(0)]),
        ])

        return loss,metrics_values,OrderedDict([])
