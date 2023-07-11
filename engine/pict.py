import numpy as np
import math
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
from utils import get_sigmod_num

import torch
from timm.utils import *
from timm.models import  model_parameters
from .iNet_cls import INetClsEngine

class PicTEngine(INetClsEngine):
    def __init__(self,config,**kwargs):
        super().__init__(config)
        
        self.thr_list = np.array([config.PICT.NOR_THR for i in range(config.PICT.INST_NUM_CLASS)])
        self.dis_ratio_list=[[] for i in range(len(self.thr_list))]

        self.train_metrics = OrderedDict([
        ('loss_cls_meter',AverageMeter()),
        ('loss_teacher_meter',AverageMeter()),
        ('dis_ins_meter',AverageMeter()),
        ('patch_num_meter',AverageMeter()),
        ('cluster_num_meter',AverageMeter()),
        ('cluster_ema_num_meter',AverageMeter()),
        ('dis_rec_meter',np.array([AverageMeter() for i in range(len(self.thr_list))])),
        ('selec_rec_meter',np.array([AverageMeter() for i in range(len(self.thr_list))]))
        ])
        self.train_metrics_epoch_log =['dis_rec_meter','selec_rec_meter']
        self.train_metrics_iter_log =['loss_cls_meter','loss_teacher_meter','dis_ins_meter','patch_num_meter','cluster_num_meter','cluster_ema_num_meter']

        self.test_metrics = OrderedDict([
        ('acc1',AverageMeter()),
        ('acc5',AverageMeter()),
        ('cluster_num_meter',AverageMeter()),
        ('auc',.0),
        ('macro_f1',.0),
        ('micro_f1',.0)
        ])
        self.test_metrics_epoch_log =['auc','macro_f1','micro_f1']
        self.test_metrics_iter_log =['acc1','acc5','patch_num_meter','cluster_num_meter']

    def __get_pseudo_label(self,config,epoch,num_steps,idx,pl_inst_shape,targets_pl,output_pl,pl_nor_cls_index):

        b,p,cls = pl_inst_shape

        t_cpu = targets_pl.cpu()
        ins_t = targets_pl.unsqueeze(-1).repeat((1,p))

        # the confidence of the category which is its bag belong
        output_bag_label =  output_pl.clone()
        output_bag_label = output_bag_label[torch.functional.F.one_hot(ins_t,num_classes=config.MODEL.NUM_CLASSES) == 1].view(b,p)
       
        out_tmp_sort,_ = torch.sort(output_bag_label,dim=-1,descending=True)         # [b p]
        min_nor_thr = out_tmp_sort[[i for i in range(b)],np.floor(p*self.thr_list[t_cpu])] # [b 1]
        min_nor_thr = min_nor_thr.unsqueeze(-1).repeat((1,p)).cuda(non_blocking=True)      # [b p]

        _,label_pl = torch.max(output_pl,dim=2)
        
        # get the index of 'normal' and 'distress' bag
        ps_mask_nor = ins_t == pl_nor_cls_index
        ps_mask_dis = ps_mask_nor==False

        # set all patch 'normal'
        if config.DATA.CLS_NOR_INDEX >=0:
            label_pl[:,:] = pl_nor_cls_index 

        # the thr of the bag
        thr_min_nor_conf = get_sigmod_num(config.PICT.THR_FIL_NOR_LOW,epoch-config.PICT.INIT_STAGE_EPOCH,config.TRAIN.EPOCHS-config.PICT.INIT_STAGE_EPOCH,end=config.PICT.THR_FIL_NOR_HIGH,alph=5)

        # RDT
        mask_ins = ps_mask_dis &  (output_bag_label >= min_nor_thr)
        label_pl[mask_ins] = ins_t[mask_ins]

        if config.PICT.FILTER_SAMPLES:
            mask_ins = (mask_ins | (ps_mask_dis & (label_pl==pl_nor_cls_index) & (output_pl[:,:,pl_nor_cls_index] >= config.PICT.THR_FIL_DIS)  )) | ((output_pl[:,:,pl_nor_cls_index] >= thr_min_nor_conf) & ps_mask_nor)
        else:
            mask_ins = label_pl == label_pl

        return label_pl,mask_ins

    def cal_loss_func(self,config,models,idx,samples,targets,epoch,num_steps,criterions,**kwargs):
        # criterions[0] is the cls loss, criterions[1] is the contr loss

        if len(criterions) == 1:
            criterions = [criterions[0],criterions[0]]
        if not isinstance(samples,(list,tuple)):
            samples = [samples,samples]

        dis_ins,p,label_pl,b,cluster_num_ema = 0,1,[],targets.size(0),[0]

        output = models['main'](samples[0])

        cluster_num = [b]
        if isinstance(output,(list,tuple)):
            if len(output) > 1:
                (output,o_inst,cluster_num) = output
            else:
                output = output[0]
                
        pl_nor_cls_index = 0 if config.PICT.INST_NUM_CLASS == 2 else config.DATA.CLS_NOR_INDEX
        
        if config.PICT.INST_NUM_CLASS != 2:
            targets_pl = targets
        else:
            if targets_bin is None:
                targets_bin = targets.clone()
                targets_bin[targets==config.DATA.DATA_NOR_INDEX] = 0
                targets_bin[targets!=config.DATA.DATA_NOR_INDEX] = 1
            targets_pl = targets_bin

        with torch.no_grad():
            _,pl_inst,cluster_num_ema = models['teacher'].module(samples[1])
            output_pl = torch.nn.functional.softmax(pl_inst,-1)
        b,p,cls = pl_inst.shape
                
        label_pl,mask_ins = self.__get_pseudo_label(config,epoch,num_steps,idx,pl_inst.shape,targets_pl,output_pl,pl_nor_cls_index)

        dis_count = torch.count_nonzero(label_pl - pl_nor_cls_index,dim=1) 
        dis_ins += torch.sum(dis_count)

        selec_rec = [ () for i in range(len(self.thr_list)) ]
        dis_rec = [ () for i in range(len(self.thr_list)) ]
        
        for i in range(len(self.thr_list)):
            i_index = targets_pl==i
            dis_tmp = dis_count[i_index]
            selec_tmp = mask_ins[i_index]

            if len(selec_tmp)>0:
                selec_rec[i] = ( torch.tensor(len(selec_tmp[selec_tmp==True]) / (len(selec_tmp)*p)),b)

            if len(dis_tmp) > 0 and epoch >= config.PICT.INIT_STAGE_EPOCH:
                thr_tmp = torch.sum(dis_tmp) / (p * len(dis_tmp))
                self.dis_ratio_list[i].append(thr_tmp.cpu())
                dis_rec[i] = (thr_tmp,b)

        label_pl = label_pl[mask_ins]
        o_inst = o_inst[mask_ins] 

        label_pl = label_pl.view(-1)
        o_inst = o_inst.view(-1,cls)

        if o_inst.size(0)>0:
            loss_pl = criterions[1](o_inst,label_pl)
        else:
            loss_pl = torch.tensor(0)


        classify_loss = criterions[0](output, targets)

        loss = config.PICT.CLASSIFY_LOSS * classify_loss + loss_pl

        metrics_values = OrderedDict([
        ('loss_cls_meter',[classify_loss,b]),
        ('loss_teacher_meter', [loss_pl,b]),
        ('dis_ins_meter',[dis_ins / (b * p),b]),
        ('patch_num_meter',[torch.tensor(len(label_pl) / (p*b)),b]),
        ('cluster_num_meter', [torch.tensor(sum(cluster_num) / b),b]),
        ('cluster_ema_num_meter',[torch.tensor(sum(cluster_num_ema)/ b),b]),
        ])
        
        metrics_values.update(OrderedDict([
            ('dis_rec_meter',dis_rec),
            ('selec_rec_meter',selec_rec)
        ]))

        return loss,metrics_values,OrderedDict([('dis_ratio_list',self.dis_ratio_list)])

    

    def update_per_iter(self,config,epoch,idx,models,**kwargs):
        if 'teacher' in models:
            teacher_ema = models['teacher']
            model = models['main']

            if config.PICT.EMA_DECAY_SCHEDULER == 'warmup' or config.PICT.EMA_DECAY_SCHEDULER == 'warmup_flat':
                teacher_ema.decay_diff = 0
            if config.PICT.EMA_DECAY_SCHEDULER == 'warmup_flat':
                teacher_ema.decay_diff = config.PICT.EMA_DECAY if epoch >= config.PICT.INIT_STAGE_EPOCH else teacher_ema.decay_diff

            teacher_ema.update(model)

    def update_per_epoch(self,config,epoch,**kwargs):
        for i in range(len(self.thr_list)):
            dis_ratio_tmp= np.sort(np.array(self.dis_ratio_list[i]))
            if len(dis_ratio_tmp) == 0:
                dis_ratio_tmp = [0]
            
            self.thr_list[i] = config.PICT.THR_REL_EMA_DECAY * self.thr_list[i] + (1-config.PICT.THR_REL_EMA_DECAY) * dis_ratio_tmp[math.floor(len(dis_ratio_tmp) * config.PICT.THR_REL_UPDATE_RATIO)]
        
        # update the dis_ratio
        self.dis_ratio_list=[[] for i in range(len(self.thr_list))]