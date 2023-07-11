import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score
import torch

from timm.utils import *
from .iNet_cls import INetClsEngine

import sys
sys.path.append("..")
from utils import getDataByStick

class PIC:
    thr=0.5 #阈值
    '''
    label = 0         #the label of image
    labels_bin = []     #0 reperents diseased，1 repernets normal
    disease_list = []    #the index list of diseased patch
    pre_label = 0      #predicted image label
    pres_bin = []      #the list of predicted patch score
    patch_num = 0       #the number of patchs
    patch_weight = []    #patch's weight
    '''
    def __init__(self,patch_num,label = 0,config=None):
        self.config = config
        self.patch_num = patch_num
        self.label = label
        self.pres_bin = np.zeros((self.patch_num,2),'float16')
        if self.label == 1:
            self.labels_bin = np.ones((patch_num,1),'int8')
            self.pre_label = np.array([0,1],'float16')
        else:
            self.labels_bin = np.zeros((patch_num,1),'int8')
            self.pre_label = np.array([1,0],'float16')

    def updateLabel_bin(self,pre = []):
        '''
        update patch label by pre
        Paras:
        pre    np.array  keras.model.predict
        _thr    float    threshold of bin-classfiy
        
        Returns:
        num    int   number of disease
        '''
        if len(pre)== 0:
            pre = self.pres_bin
        thr = PIC.thr
        num_changed = 0
        self.disease_list = []
        self.pres_bin = pre
        pre_max = 0
        tem = np.hsplit(pre,2)
        sorted_pre = tem[1]
        sorted_pre = np.sort(sorted_pre,axis = 0)
        if self.label == 1:
            for i in range(0,len(pre)):
                if pre_max < pre[i][1]:
                    pre_max = pre[i][1]
                if pre[i][1] > thr and self.labels_bin[i] == 0:
                    self.labels_bin[i] = 1
                    self.disease_list.append(i)
                    num_changed = num_changed +1
                elif pre[i][1]<= thr and self.labels_bin[i] == 1 and pre[i][1]<= sorted_pre[(int)(len(sorted_pre) * self.config.IOPLIN.R_THR_RATIO)]:
                    self.labels_bin[i] = 0
                    num_changed = num_changed +1
                else:
                    if self.labels_bin[i] == 1:
                        self.disease_list.append(i)
                               
        return len(self.disease_list),num_changed
    
    def preNor(self,pre = []):
        '''
        according to the thrshold to detect whether the image is normal
        '''
        if len(pre)== 0:
            pre = self.pres_bin
        thr = PIC.thr
        num_dis = 0
        pre_max = 0
        for i in range(0,len(pre)):
            if pre_max < pre[i][1]:
                pre_max = pre[i][1]
            if pre[i][1] > thr:
                num_dis = num_dis + 1
        self.pre_label[1] = pre_max
        self.pre_label[0] = 1 - self.pre_label[1] 
        if num_dis < 1:
            return True
        else :
            return False
        
    def getSampleWeight(self,pre = []):  
        '''
        return patch's weight
        Paras:
        thr     float     thrshold of binary classification
        pre     list      patch's predicted score
        '''
        thr = PIC.thr
        if len(pre)== 0:
            pre = self.pres_bin
        s_weight = []
        for i in range(0,len(self.labels_bin)):
            tem = 1 * (pre[i][1] / thr)
            if tem < self.config.IOPLIN.MIN_SAMPLE_WEIGHT:                
                tem = self.config.IOPLIN.MIN_SAMPLE_WEIGHT
            s_weight.append(tem)
        self.patch_weight = s_weight
        self.labels_bin_bfLast = self.labels_bin
        return s_weight
        
    def cvtData(self,x = [],y = [],is_x=True,is_y=True,is_del = True):
        '''
        put the patch file to the external variable
        Paras:
        x      list     external x
        y      list     external y
        is_x    bool     whether process x
        is_y    bool     whether process y
        is_del   bool     whether delete the inner image 
        '''
        num = len(self.labels_bin)
        if is_x:
            for i in range(0,num):
                x.append(np.tile(self.pics[i]),(1,1,3))
            if is_del:
                del self.pics
        if is_y:
            for j in range(0,num):
                y.append(self.labels_bin[j])

class IOPLINEngine(INetClsEngine):
    def __init__(
        self,
        config,
        **kwargs):
        super().__init__(config)

        self.pics_train = np.array(['' for i in range(config.DATA.LEN_DATASET_TRAIN)],dtype=object)
        self.pics_val = np.array(['' for i in range(config.DATA.LEN_DATASET_VAL)],dtype=object)
        self.pics_test = np.array(['' for i in range(config.DATA.LEN_DATASET_TEST)],dtype=object)
        self.random_indi = []
        self.indi_start=0
        self.device = 'cuda:'+str(config.LOCAL_RANK) if torch.cuda.is_available() else 'cpu'

        self.pr_stick = [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
        self._binary_test = config.MODEL.NUM_CLASSES == 2 or config.TEST.BINARY_MODE 

        self.train_metrics = OrderedDict([
            ('cls_loss',AverageMeter())
        ])
        self.train_metrics_epoch_log =['thr','cls_loss','num_dis','num_patch']
        self.train_metrics_iter_log =['cls_loss']

        if self._binary_test:
        # 添加p@r评价指标
            self.test_metrics.update(OrderedDict([
                    ('p@r90',.0),
                    ('p@r95',.0),
                    ('p@r99',.0),
            ]))
            self.test_metrics_epoch_log += ['p@r90','p@r95','p@r99']

    def cal_loss_func(self,config,models,idx,samples,targets,epoch,num_steps,criterions,**kwargs):
        bs = targets.size(0)
        ori_reduction = criterions[0].reduction
        if epoch==0:
            targets = torch.flatten(targets.view(-1,1).repeat(1,config.DATA.NUM_PATCHES))
        else:
            target=[]
            weight = []
            for i in range(bs):
                target.append((self.pics_train[self.random_indi[self.indi_start+i]].labels_bin))
                weight.append((self.pics_train[self.random_indi[self.indi_start+i]].patch_weight))
            del targets
            targets = torch.LongTensor(target).cuda()
            weight = torch.flatten(torch.Tensor(weight).cuda())
            self.indi_start += bs
            targets = torch.flatten(targets)
            
            criterions[0].reduction = 'none'

        predictions = models['main'](samples)
        cls_loss = criterions[0](predictions,targets)
        if epoch != 0:
            cls_loss = cls_loss * weight
            cls_loss = torch.mean(cls_loss)
        loss = cls_loss 
        metrics_values = OrderedDict([
            ('cls_loss',[cls_loss,targets.size(0)]),
        ])
        criterions[0].reduction = ori_reduction
        return loss,metrics_values,OrderedDict([])

    def update_before_train(self,config,models,epoch,dataloader,**kwargs):
        self.random_indi = np.arange(config.DATA.LEN_DATASET_TRAIN)
        
        np.random.shuffle(self.random_indi)
        dataloader.sampler.set_indices(self.random_indi)
        self.indi_start = 0

    def update_pesudo_label(self,pics,config,models,dataloader):
        is_timm_loader = config.DATA.DATALOADER_NAME.startswith('timm')
        num_dis = 0
        indi_start = 0

        for idx, (samples, targets) in enumerate(dataloader):
            # timm dataloader prefetcher will do this
            if not is_timm_loader or not config.DATA.TIMM_PREFETCHER:
                if type(samples) in (tuple,list):
                    for _i in range(len(samples)):
                        samples[_i] = samples[_i].cuda(non_blocking=config.DATA.PIN_MEMORY)
                else:
                    samples = samples.cuda(non_blocking=config.DATA.PIN_MEMORY)
                targets = targets.cuda(non_blocking=True)

            with torch.no_grad():
                bs = targets.size(0)
                output = models['main'](samples)
                output_soft = torch.nn.functional.softmax(output,dim=1)
                output_soft=output_soft.view(targets.size(0),config.DATA.NUM_PATCHES,config.MODEL.NUM_CLASSES)
                if pics is not None:
                    if pics[self.random_indi[indi_start]]=='':
                        for i in range(bs):
                            pics[self.random_indi[indi_start+i]] = PIC(patch_num=config.DATA.NUM_PATCHES,label=targets[i].cpu().numpy(),config=config)
                    #获取当前的包预测值
                    for i in range(bs):
                        #更新标签
                        num_dis_p,_ = pics[self.random_indi[indi_start+i]].updateLabel_bin(pre=output_soft[i].cpu().numpy())
                        num_dis += num_dis_p
                        pics[self.random_indi[indi_start+i]].preNor(pre=output_soft[i].cpu().numpy())
                        pics[self.random_indi[indi_start+i]].getSampleWeight(pre=output_soft[i].cpu().numpy())
                    indi_start += bs
        return num_dis

    def update_per_epoch(self,config,epoch,models,dataloader,**kwargs):
        if epoch % config.IOPLIN.UPDATE_FREQ == 0:
            num_dis = self.update_pesudo_label(self.pics_train,config,models,dataloader)
            PIC.thr = num_dis / (len(dataloader)*config.DATA.BATCH_SIZE*config.DATA.NUM_PATCHES) if num_dis!=0 else PIC.thr
            self.indi_start = 0
            return OrderedDict([
                ('thr',PIC.thr),
                ('num_dis',num_dis),
                ('num_patch',len(dataloader)*config.DATA.BATCH_SIZE*config.DATA.NUM_PATCHES)
            ])
        else:
            self.indi_start = 0

    def measure_per_iter(self,config,models,samples,targets,criterions,is_test=True,**kwargs):
        bs = targets.size(0)

        # compute output
        output = models['main'](samples)

        if isinstance(output, (tuple, list)):
            predition = output[0]
        else:
            predition = output

        output_soft = torch.nn.functional.softmax(predition,dim=-1)

        output_soft=output_soft.view(targets.size(0),config.DATA.NUM_PATCHES,config.MODEL.NUM_CLASSES)
        output_i = []
        pics = self.pics_test if is_test else self.pics_val
        if pics is not None:
            if pics[self.indi_start]=='':
                for i in range(bs):
                    pics[self.indi_start+i] = PIC(patch_num=config.DATA.NUM_PATCHES,label=targets[i].cpu().numpy())
            #获取当前的包预测值
            for i in range(bs):
                pics[self.indi_start+i].preNor(pre=output_soft[i].cpu().numpy())
                output_i.append(pics[self.indi_start+i].pre_label)
            self.indi_start += bs
        output_i = torch.tensor(output_i).cuda()
        del output
        output_soft = output_i

        loss = criterions[0](output_soft, targets)
            
        pred = output_soft.cpu().numpy()
        label = targets.cpu().numpy()

        # topk acc cls
        acc1,acc5 = accuracy(output_soft, targets, topk=self.topk)

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
            try:
                precision,recall,thr=precision_recall_curve(label, pred)
                patr=getDataByStick([precision,recall],self.pr_stick)
                auc = roc_auc_score(label,pred)
                metrics_values.update(OrderedDict([
                    ('auc',auc),
                    # ('p@r90',patr[2][0]),
                    ('p@r95',patr[1][0]),
                    ('p@r99',patr[0][0]),
                ]))
            except:
                metrics_values.update(OrderedDict([
                    ('auc',auc),
                    # ('p@r90',0),
                    ('p@r95',0),
                    ('p@r99',0),
                ]))

        metrics_values.update(OrderedDict([
        ('macro_f1',ma_f1),
        ('micro_f1',mi_f1)
        ]))
        others = OrderedDict([
        ])

        self.indi_start = 0
        
        return metrics_values,others
