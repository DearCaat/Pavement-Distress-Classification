from collections import OrderedDict
import csv
import os
import torch
import torch.distributed as dist
import shutil
from copy import deepcopy
import math
import torch.nn.functional as F
import torch.nn as nn
if torch.__version__ >= '2.0':
    from torch import inf
else:
    from torch._six import inf
import numpy as np
import wandb

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def del_ckp_model(config,is_ema=False):
    ema_prefix = '_ema' if is_ema else ''

    for best_model_name in config.MODEL.SAVE_BEST_MODEL_NAME:
        prefix = best_model_name
        ckpt_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f"_{config.EXP_NAME}"+f"_{prefix}"+f"{ema_prefix}"+'_ckpt.pth')
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

def load_best_model_V2(config,models,logger,is_ema=False):
    ema_prefix = '_ema' if is_ema else ''

    for best_model_name in config.MODEL.SAVE_BEST_MODEL_NAME:
        prefix = best_model_name
        ckpt_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f"_{config.EXP_NAME}"+f"_{prefix}"+f"{ema_prefix}"+'_ckpt.pth')
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f"_{config.EXP_NAME}"+f"_{prefix}"+f"{ema_prefix}"+'_btml.pth')

        logger.info(f"==============> Loading the best {prefix} {ema_prefix} model....................")
        checkpoint = torch.load(best_path, map_location='cpu')

        if 'epoch' in checkpoint:
            logger.info(f"==============> Epoch {checkpoint['epoch']}....................")

        if best_model_name == 'main':
            msg = models[best_model_name].load_state_dict(checkpoint['state_dict'], strict=False)
            if config.APEX_AMP and checkpoint['config'].APEX_AMP:
                amp.initialize(models['main'], opt_level='O1')
        else:
            msg = models[best_model_name].load_state_dict(checkpoint['other_models'][best_model_name], strict=False)

        logger.info(f"{prefix}: {msg}")

        if is_ema:
            return

def load_best_model(config,models,logger,is_ema=False):

    if is_ema:
        ckpt_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_ckpt.pth')
    else:
        ckpt_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ckpt.pth')
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if is_ema:
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_best_model.pth')
    else:
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_best_model.pth')
    logger.info(f"==============> Loading the best model....................")
    checkpoint = torch.load(best_path, map_location='cpu')
    if 'epoch' in checkpoint:
        logger.info(f"==============> Epoch {checkpoint['epoch']}....................")
    msg = models['main'].load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(msg)
    if config.APEX_AMP and checkpoint['config'].APEX_AMP:
        amp.initialize(models['main'], opt_level='O1')

def load_checkpoint(config, model,optimizer=None, lr_scheduler=None,logger=None,is_ema=False):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if is_ema:
        if 'ema' in checkpoint:
            msg = model.load_state_dict(checkpoint['ema'], strict=False)
            logger.info(msg)
            return 0
    # 是否只读取模型
    if 'state_dict' in checkpoint:
        msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(msg)
        max_accuracy = 0.0
        best_auc = 0.0
        best_f1 = 0.0
        if config.TRAIN_MODE=='train' or config.TRAIN_MODE=='t_e' :
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                config.defrost()
                config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
                config.freeze()
                if 'amp' in checkpoint and config.APEX_AMP and checkpoint['config'].APEX_AMP:
                    amp.initialize(model, opt_level='O1')
                    amp.load_state_dict(checkpoint['amp'])
                logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
                if 'max_accuracy' in checkpoint and 'best_auc' in checkpoint:
                    max_accuracy = checkpoint['max_accuracy']
                    best_auc = checkpoint['best_auc']
                if 'best_f1' in checkpoint:
                    best_f1 = checkpoint['best_f1']
    else:
        msg = model.load_state_dict(checkpoint, strict=False)
        logger.info(msg)
    del checkpoint
    return max_accuracy,best_auc,best_f1

def load_checkpoint_V2(config, models,optimizer=None, lr_scheduler=None,logger=None,model_ema=None):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        try:
            checkpoint = torch.load(config.MODEL.RESUME, map_location = 'cpu')
        except:
            checkpoint = torch.load(os.path.join(config.OUTPUT,'model',config.MODEL.RESUME), map_location = 'cpu')
    
    # 是否只读取模型
    if 'state_dict' in checkpoint and not config.MODEL.ONLY_LOAD_MODEL:
        config.defrost()
        msg = models['main'].load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(f"main: {msg}")
        if 'ema' in checkpoint and model_ema is not None:
            msg = model_ema.load_state_dict(checkpoint['ema'], strict=False)
            logger.info(f"ema: {msg}")

        if 'other_models' in checkpoint:
            other_models = checkpoint['other_models']
            config.MODEL.SAVE_OTHER_MODEL_NAME = list(other_models.keys())
            for model_prefix in config.MODEL.SAVE_OTHER_MODEL_NAME:
                msg = models[model_prefix].load_state_dict(other_models[model_prefix])
                logger.info(f"{model_prefix}: {msg}")

        best_metrics = None
        best_metrics_ema = None

        if config.TRAIN_MODE=='train' or config.TRAIN_MODE=='t_e':
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
                if 'amp' in checkpoint and config.APEX_AMP and checkpoint['config'].APEX_AMP:
                    amp.initialize(models['main'], opt_level='O1')
                    amp.load_state_dict(checkpoint['amp'])
                logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
                if 'best_metrics' in checkpoint:
                    best_metrics = checkpoint['best_metrics']
                if 'best_metrics_ema' in checkpoint:
                    best_metrics_ema = checkpoint['best_metrics_ema']
            config.freeze()
    else:
        msg = models['main'].load_state_dict(checkpoint, strict=False)
        logger.info(msg)
    del checkpoint
    return best_metrics,best_metrics_ema

def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger,is_best,best_auc,best_f1,ema,is_ema=False,best_patr90=0.0):
    save_state = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'max_accuracy': max_accuracy,
                  'best_auc': best_auc,
                  'best_f1': best_f1,
                  'p@r90':best_patr90,
                  'epoch': epoch,
                  'config': config,
                  'ema':ema.module.state_dict() if ema is not None else None}
    save_state_best = {'state_dict': model.state_dict(),
                        'max_accuracy': max_accuracy,
                        'best_auc': best_auc,
                        'best_f1': best_f1,
                        'p@r90':best_patr90,
                        'epoch': epoch,
                        'config': config}
    if config.TRAIN.LR_SCHEDULER.NAME is not None:
        save_state['lr_scheduler'] = lr_scheduler.state_dict()
    if config.APEX_AMP:
        amp.initialize(model, opt_level='O1')
        save_state['amp'] = amp.state_dict()

    if is_ema:
        save_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_ckpt.pth')
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ema_best_model.pth')
        history_best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_his_ema_best_model.pth')
    else:
        save_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_ckpt.pth')
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+config.EXP_NAME+f'_best_model.pth')
        history_best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f'_his_best_model.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    
    if is_best:
        torch.save(save_state_best, best_path)
        # shutil.copyfile(save_path, best_path)
    if epoch == config.TRAIN.EPOCHS - 1:
        if os.path.exists(history_best_path):
            checkpoint = torch.load(best_path, map_location='cpu')
            checkpoint_his = torch.load(history_best_path, map_location='cpu')
            if config.TEST.BEST_MODEL_METRIC.lower() == 'f1':
                if 'best_f1' in checkpoint_his:
                    if checkpoint['best_f1'] > checkpoint_his['best_f1']:
                        shutil.copyfile(best_path, history_best_path)
                else:
                    shutil.copyfile(best_path, history_best_path)
            elif config.TEST.BEST_MODEL_METRIC.lower() == 'p@r90':
                if 'p@r90' in checkpoint_his:
                    if checkpoint['p@r90'] > checkpoint_his['p@r90']:
                        shutil.copyfile(best_path, history_best_path)
                else:
                    shutil.copyfile(best_path, history_best_path)
            elif config.TEST.BEST_MODEL_METRIC.lower() == 'top1':
                if checkpoint['max_accuracy'] > checkpoint_his['max_accuracy']:
                    shutil.copyfile(best_path, history_best_path)
            elif config.TEST.BEST_MODEL_METRIC.lower() == 'auc':
                if checkpoint['best_auc'] > checkpoint_his['best_auc']:
                    shutil.copyfile(best_path, history_best_path)
        else:
            shutil.copyfile(best_path, history_best_path)

    logger.info(f"{save_path} saved !!!")


def _save_checkpoint_V2(config, epoch, models, best_metrics,optimizer, lr_scheduler, logger,is_best,ema,prefix='',best_model_name='main',is_ema=False,best_metrics_ema=None,is_last=False):
    ema_prefix = '_ema' if is_ema else ''
    other_models_sd = {}
    for m in config.MODEL.SAVE_OTHER_MODEL_NAME:
        if str(m).lower() == 'main':
            continue
        else:
            other_models_sd[str(m)] = models[m].state_dict()
    save_state = {'state_dict': models['main'].state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'best_metrics':best_metrics,
                  'epoch': epoch,
                  'config': config,
                  'other_models':other_models_sd,
                  'ema':ema.module.state_dict() if ema is not None else None,
                  'best_metrics_ema': best_metrics_ema}
    # 全部保存占用空间太大，最佳模型只保存部分模型信息              
    best_state = {
        'state_dict': ema.module.state_dict() if is_ema else models['main'].state_dict(),
        'config': config,
        'epoch': epoch,
        'best_metrics': best_metrics_ema if is_ema else best_metrics,
    }

    if config.TRAIN.LR_SCHEDULER.NAME is not None:
        save_state['lr_scheduler'] = lr_scheduler.state_dict()
    if config.APEX_AMP:
        amp.initialize(models['main'], opt_level='O1')
        save_state['amp'] = amp.state_dict()

    # get the saved filename
    save_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f"_{config.EXP_NAME}"+f"_{prefix}"+f"{ema_prefix}"+'_ckpt.pth')

    logger.info(f"{save_path} saving......")
    try:
        torch.save(save_state, save_path)
    except Exception as e:
        logger.info(f"{repr(e)}")

    # save the last model, if needed
    if is_last:
        logger.info(f"Last Epoch model: "+config.MODEL.NAME+f"_{config.EXP_NAME}"+f"_{prefix}"+f"{ema_prefix}"+'_ltml.pth')
        last_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f"_{config.EXP_NAME}"+f"_{prefix}"+f"{ema_prefix}"+'_ltml.pth')
        try:
            torch.save(best_state, last_path)
            logger.info(f"Success!")
        except Exception as e:
            logger.info(f"{repr(e)}")

    if is_best:
        logger.info(f"Best model: "+config.MODEL.NAME+f"_{config.EXP_NAME}"+f"_{prefix}"+f"{ema_prefix}"+'_btml.pth')
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f"_{config.EXP_NAME}"+f"_{prefix}"+f"{ema_prefix}"+'_btml.pth')
        try:
            torch.save(best_state, best_path)
            logger.info(f"Success!")
        except Exception as e:
            logger.info(f"{repr(e)}")

        # shutil.copyfile(save_path, best_path)
    if epoch == config.TRAIN.EPOCHS - 1 and not config.TRAIN.NO_VAL:
        # 多次实验留下的最佳
        best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f"_{config.EXP_NAME}"+f"_{prefix}"+f"{ema_prefix}"+'_btml.pth')
        history_best_path = os.path.join(config.OUTPUT, 'model',config.MODEL.NAME+f"_{prefix}"+f"{ema_prefix}"+'_hbtml.pth')
        best_model_metirc = list2dict(config.TEST.BEST_MODEL_METRIC)
        if os.path.exists(history_best_path):
            checkpoint = torch.load(best_path, map_location='cpu')
            checkpoint_his = torch.load(history_best_path, map_location='cpu')

            metrics = checkpoint['best_metrics']
            metrics_his = checkpoint_his['best_metrics']
            best_metric_name = best_model_metirc[str(best_model_name)].lower()

            if best_metric_name in metrics_his:
                if metrics[best_metric_name] > metrics_his[best_metric_name]:
                    try:
                        shutil.copyfile(best_path, history_best_path)
                    except Exception as e:
                        logger.info(f"{repr(e)}")
            else:
                try:
                    shutil.copyfile(best_path, history_best_path)
                except Exception as e:
                        logger.info(f"{repr(e)}")
        else:
            try:
                shutil.copyfile(best_path, history_best_path)
            except Exception as e:
                    logger.info(f"{repr(e)}")

    logger.info(f"{save_path} saved !!!")

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
    
def l1_regularizer(input_features):
    l1_loss = torch.norm(input_features, p=1)
    return l1_loss

def l2_regularizer(input_features):
    l2_loss = torch.norm(input_features, p=2)
    return l2_loss

def decimal_num(number):
    if type(number) == int:
        return 0
    else:
        num = 1
        while number * 10 ** num != int(number * 10 ** num ):
            num += 1
        return num

def list2dict(_list):
    # [key,value,key2,value2,...] --> {key:value,key2:value2}
    assert len(_list) % 2 == 0
    _dict = {}
    for i in range(0,len(_list),2):
        _dict.update({_list[i]:_list[i+1]})
    return _dict
   
def getDataByStick(data,stick):
    '''
    data    [precision,recall]   y坐标数据，x坐标数据
    stick                     坐标刻度
    '''
    j = -1
    diff = 9999
    _stick = stick.copy()
    _list = []
    for a in range(0,len(data[0])):
        if len(_stick):
            diff_tem = abs(data[1][a]-_stick[j])
            if diff_tem < diff:
                diff = diff_tem
            if diff_tem > diff:
                #print(data[0][a-1],data[1][a-1])
                _list.append((data[0][a-1],data[1][a-1]))
                if len(_stick)>1:
                    diff = abs(data[1][a]-_stick[j-1])
                else:
                    diff=9999
                del _stick[j]
    return _list

def get_sigmod_num(start=0,curr_step=0,all_step=0,end=0.999,alph=10):
    '''
    alph  平缓系数,数字越小则曲线越平缓
    '''
    thr_min_conf = start + (round((1 / (1+math.exp(-alph*(curr_step / all_step)))-0.5 )*2,3)) * (end-start)
    return thr_min_conf

class ModelEmaV3(torch.nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, decay_diff=0.9999,device=None,diff_layers = [],ban_para = [],init_para=[],momentum_schedule=None):
        super(ModelEmaV3, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.decay_diff = decay_diff
        self.diff_layers = diff_layers
        self.ban_para = ban_para
        self.device = device  # perform ema on different device from model if set
        self.momentum_schedule = momentum_schedule
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for k,ema_v, model_v in zip(self.module.state_dict().keys(),self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                if k.split('.')[0] not in self.ban_para:
                    if k.split('.')[0] in self.diff_layers:
                        ema_v.copy_(update_fn(ema_v, model_v,self.decay_diff))
                    else:
                        ema_v.copy_(update_fn(ema_v, model_v,self.decay))

    def update(self, model,iter=None):
        if self.momentum_schedule is not None:
            self.decay = self.momentum_schedule[iter]
        self._update(model, update_fn=lambda e, m, decay: decay * e + (1. - decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def log_loss(tea,stu,config):
    tps_stu = 1 if config.PICT.SHARPEN_STUDENT is None else config.PICT.SHARPEN_STUDENT
    tps_tea = 1 if config.PICT.SHARPEN_TEACHER is None else config.PICT.SHARPEN_TEACHER
    tea = tea.detach()
    stu =  stu / tps_stu
    tea = torch.nn.functional.softmax(tea / tps_tea,dim=-1)
    return -(tea*F.log_softmax(stu,dim=-1).sum(dim=-1).mean())

# 相较于timm的版本，我在这里对target也做softmax
class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy_v2, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target,dim=-1) * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

# ref: https://github.com/microsoft/Swin-Transformer/blob/3b0685bf2b99b4cf5770e47260c0f0118e6ff1bb/utils.py#L195
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

# 相较于timm的版本，我想要实现gradient accumulation，需要把梯度计算和更新参数步骤分开
# ref: https://github.com/microsoft/Swin-Transformer/blob/3b0685bf2b99b4cf5770e47260c0f0118e6ff1bb/utils.py#L195
class NativeScaler_V2:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer=None, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False,update_grad=False,change_grad_func=None,**kwargs):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                # dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            change_grad_func(**kwargs)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

# 与timm的主要区别在于，timm没有对两个metrics做None值判断
def update_summaryV2(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False,eval_metrics_ema=None):
    rowd = OrderedDict(epoch=epoch)
    if train_metrics is not None:
        rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if eval_metrics is not None:
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if eval_metrics_ema is not None:
        rowd.update([('eval_ema_' + k, v) for k, v in eval_metrics_ema.items()])
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

# 处理多折测试时的评价标准
def update_kfold_metrics(config,metrics_kfold,metrics):
    is_end = config.TEST.K_FOLD_VAL_NOW == config.TEST.K_FOLD_VAL_TIMES

    metrics_kfold.append(metrics)
    metrics_tmp = deepcopy(metrics)

    # 最后一折
    if is_end:
        metrics.update([('final/'+k+'_mean',np.mean(np.array([ m_kf[k] for m_kf in metrics_kfold]))) for k, v in metrics_tmp.items()])
        metrics.update([('final/'+k+'_std',np.std(np.array([m_kf[k] for m_kf in metrics_kfold]))) for k, v in metrics_tmp.items()])
    else:
        metrics = OrderedDict([ (str(config.TEST.K_FOLD_VAL_NOW)+'-fold/'+k,v) for k, v in metrics.items()])
    return metrics

