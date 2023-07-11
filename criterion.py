import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import numpy as np

def _build_criterion(name,config):
    if name == 'crossentropy':
        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    elif name == 'kl':
        kl_para = {
            'out_dim': config.TRAIN.LOSS.FEATURE_NUM,
            's_temp': config.TRAIN.LOSS.CL_STUDENT_TEMP if hasattr(config.TRAIN.LOSS, 'CL_STUDENT_TEMP') else 1,
            't_temp': config.TRAIN.LOSS.CL_TEACHER_TEMP if hasattr(config.TRAIN.LOSS, 'CL_TEACHER_TEMP') else 1,
            'center_momentum': config.TRAIN.LOSS.CENTER_MOMENTUM if hasattr(config.TRAIN.LOSS, 'CENTER_MOMENTUM') else None,
            'is_dist': config.DISTRIBUTED
        }
        criterion = SoftTargetCrossEntropy_v2(**kl_para)
    elif name == 'dino':
        criterion = DINOLoss(config.DINO.OUT_DIM,config.DINO.LOCAL_CROPS_NUMBER+2,config.DINO.WARMUP_TEACHER_TEMP,config.DINO.TEACHER_TEMP,config.DINO.WARMUP_TEACHER_TEMP_EPOCHS,config.TRAIN.EPOCHS)
    else:
        raise NotImplementedError
    return criterion

def build_criterion(config):
    '''
    loss_name : XX_XX_XX
    '''
    criterions = []
    losses_name = config.TRAIN.LOSS.NAME.lower().split('_')

    for loss_name in losses_name:
        criterions.append(_build_criterion(loss_name,config))

    return criterions

def log_loss(tea,stu,config):
    tps_stu = 1 if config.PICT.SHARPEN_STUDENT is None else config.PICT.SHARPEN_STUDENT
    tps_tea = 1 if config.PICT.SHARPEN_TEACHER is None else config.PICT.SHARPEN_TEACHER
    tea = tea.detach()
    stu =  stu / tps_stu
    tea = torch.nn.functional.softmax(tea / tps_tea,dim=-1)
    return -(tea*F.log_softmax(stu,dim=-1).sum(dim=-1).mean())

# 相较于timm的版本，我在这里对target也做softmax
class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self,out_dim,s_temp=1.,t_temp=1.,center_momentum=None,is_dist=False):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.s_temp = s_temp
        self.t_temp = t_temp
        self.center_momentum = center_momentum
        self.is_dist = is_dist
        if center_momentum is not None:
            self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if self.is_dist:
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        if self.center_momentum is not None:
            loss = torch.sum(-F.softmax((target - self.center)/self.t_temp,dim=-1) * F.log_softmax(x/self.s_temp, dim=-1), dim=-1)
        else:
            loss = torch.sum(-F.softmax(target/self.t_temp,dim=-1) * F.log_softmax(x/self.s_temp, dim=-1), dim=-1)

        loss = loss.mean()
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()

        if self.center_momentum is not None:
            self.update_center(target)

        return loss

# copyright dino@facebook,ref: https://github.com/facebookresearch/dino/blob/main/main_dino.py
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9,is_dist=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.is_dist = is_dist
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if self.is_dist:
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)