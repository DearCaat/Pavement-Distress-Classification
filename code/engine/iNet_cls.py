import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
import torch
from timm.utils import *

import sys
sys.path.append("..")
from utils import getDataByStick

class INetClsEngine:
    def __init__(self,config,**kwargs):
        # 除了主损失以外的metric，主损失会每个iter进行log
        # 每个iter更新的指标需要初始化成AverageMeter
        # 默认情况下，训练函数内不进行任何评价指标计算
        self.train_metrics = OrderedDict([])
        self.train_metrics_epoch_log =[]
        self.train_metrics_iter_log =[]

        self.topk = config.TEST.TOPK if config.MODEL.NUM_CLASSES > max(config.TEST.TOPK) else (1,max(config.TEST.TOPK))

        self.test_metrics = OrderedDict([
        ('macro_f1',.0),
        ('micro_f1',.0),])

        self.test_metrics.update(OrderedDict([
            ('acc'+str(i),AverageMeter()) for i in self.topk
        ]))

        self.test_metrics_epoch_log =['macro_f1','micro_f1']
        self.test_metrics_iter_log =['acc'+str(i) for i in self.topk]

        if config.TEST.BINARY_MODE or config.MODEL.NUM_CLASSES == 2:
            self.test_metrics.update(OrderedDict([
                ('auc',.0),
            ]))
            self.test_metrics_epoch_log += ['auc']

            if config.TEST.PR:
                self.pr_stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
                # 添加p@r评价指标
                self.test_metrics.update(OrderedDict([
                        ('p@r90',.0),
                        ('p@r95',.0),
                        ('p@r99',.0),
                ]))
                self.test_metrics_epoch_log += ['p@r90','p@r95','p@r99']
        
    def cal_loss_func(self,config,models,idx,samples,targets,epoch,num_steps,criterions,**kwargs):
        if isinstance(samples,(tuple,list)):
            samples = samples[0]
        predictions = models['main'](samples)
        loss = criterions[0](predictions,targets)
        metrics_values = OrderedDict([
        ])

        return loss,metrics_values,OrderedDict([])

    def update_before_train(self,config,epoch,models,dataloader,**kwargs):
        return
    def update_per_iter(self,config,epoch,idx,models=None,**kwargs):
        return

    def update_per_epoch(self,config,epoch,models=None,**kwargs):
        return

    # do something to gradients, this function is call after cal the gradients(include the clip gradients), and before the optimize
    def change_grad_func(self,config,epoch,models,idx,**kwargs):
        return

    def measure_per_iter(self,config,models,samples,targets,criterions,**kwargs):
        # compute output
        if isinstance(samples,(tuple,list)):
            samples = samples[0]
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
        acck = accuracy(predition, targets, topk=self.topk)

        metrics_values = OrderedDict([
            ('acc'+str(self.topk[i]),[acck[i],targets.size(0)]) for i in range(len(self.topk))
        ])

        others = OrderedDict([])

        return loss,pred,label,metrics_values,others
        
    def measure_per_epoch(self,config,**kwargs):
        assert not(config.MODEL.NUM_CLASSES == 2 and config.TEST.BINARY_MODE)

        metrics_values = OrderedDict([])
        
        _binary_test = config.MODEL.NUM_CLASSES == 2 or config.TEST.BINARY_MODE
        label = kwargs['label']
        pred = kwargs['pred']

        ma_f1 = f1_score(label,np.argmax(pred,axis=1),average='macro')
        mi_f1 = f1_score(label,np.argmax(pred,axis=1),average='micro')

        if _binary_test:
            if config.TEST.BINARY_MODE:
                label = label!=config.DATA.DATA_NOR_INDEX
                pred = 1-pred[:,config.DATA.DATA_NOR_INDEX]
            elif config.MODEL.NUM_CLASSES == 2:
                ma_f1 = f1_score(label,np.argmax(pred,axis=1),average='binary')
                mi_f1 = ma_f1
                pred = 1-pred[:,config.DATA.CLS_NOR_INDEX]

            # print(label)
            try:
                auc = roc_auc_score(label,pred)
            except:
                auc = 0.
            try:
                if config.TEST.PR:
                    precision,recall,thr=precision_recall_curve(label, pred)
                    patr=getDataByStick([precision,recall],self.pr_stick)
                    metrics_values.update(OrderedDict([
                        ('p@r90',patr[2][0]),
                        ('p@r95',patr[1][0]),
                        ('p@r99',patr[0][0]),
                    ]))
            except:
                metrics_values.update(OrderedDict([
                        ('p@r90',0),
                        ('p@r95',0),
                        ('p@r99',0),
                    ]))
            metrics_values.update(OrderedDict([
                ('auc',auc),
            ]))
            

        metrics_values.update(OrderedDict([
        ('macro_f1',ma_f1),
        ('micro_f1',mi_f1)
        ]))
        others = OrderedDict([
        ])

        return metrics_values,others
