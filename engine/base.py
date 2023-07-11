from contextlib import suppress
from copy import deepcopy
import time
import numpy as np
from numpy import ndarray
import datetime
from collections import OrderedDict

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

import sys,os 
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils import ampscaler_get_grad_norm

import torch
from torch import Tensor
from timm.utils import *
from timm.models import  model_parameters

def _update_list_meter(config,metrics_list,value_list,distributed=False):
    for metric,value in zip(metrics_list,value_list):
        if isinstance(metric,(list,tuple,Tensor,ndarray)):
            _update_list_meter(config,metric,value,distributed)
        elif isinstance(metric,AverageMeter):
            if len(value) == 0:
                continue
            if distributed:
                value[0] = reduce_tensor(value[0],config.WORLD_SIZE)
            metric.update(value[0].item(),value[1])
        else:
            metric = value
def update_metrics(config,metrics,values,distributed=False):
    for key,value in values.items():
        if isinstance(metrics[key],(list,tuple,Tensor,ndarray)):
            _update_list_meter(config,metrics[key],value,distributed)
        elif isinstance(metrics[key],AverageMeter):
            if distributed and isinstance(value[0],torch.Tensor):
                value[0] = reduce_tensor(value[0],config.WORLD_SIZE)
            _value = value[0].item() if isinstance(value[0],torch.Tensor) else value[0]
            metrics[key].update(_value,value[1])
        else:
            metrics[key] = value

def _log_list_meter(_list,key,logger):
    log_str = ''
    for idx,item in enumerate(_list):
        if isinstance(item,(list,tuple,Tensor,ndarray)):
            _log_list_meter(*item)
        elif isinstance(item,AverageMeter):
            log_str += f'{key}_{idx} {item.val:.4f} ({item.avg:.4f})\t'
        else:
            log_str += f'{key}_{idx} {item:.4f}\t'
    logger.info(log_str)

def log_meter(metrics,log_list,logger):
        log_str = ''
        wandb_dic = OrderedDict([])
        for key, value in metrics.items():
            if key in log_list:
                key = key[:-6] if '_meter' in key else key
                if isinstance(value,(list,tuple,Tensor,ndarray)):
                    _log_list_meter(value,key,logger)
                elif isinstance(value,AverageMeter):
                    log_str += f'{key} {value.val:.4f} ({value.avg:.4f})\t'
                    wandb_dic.update(OrderedDict([
                        (key,value.avg)
                    ]))
                else:
                    log_str += f'{key} {value:.4f}\t'
                    wandb_dic.update(OrderedDict([
                        (key,value)
                    ]))
        if len(log_str) != 0:
            logger.info(log_str)

        return wandb_dic

def reset_meter(metrics):
    for _key in metrics.keys():
        if isinstance(metrics[_key],AverageMeter):
            metrics[_key].reset()

class BaseTrainer():
    def __init__(self,engine,config,**kwargs):
        self.engine = engine
        
        if config.TRAIN.NO_VAL:
            self.best_metrics = deepcopy(engine.train_metrics)
            self.best_metrics.update(OrderedDict([
                ('loss',0)
            ]))
        else:
            self.best_metrics = deepcopy(engine.test_metrics)
        for key, value in self.best_metrics.items():
            if isinstance(value,AverageMeter):
                self.best_metrics[key] = value.avg

    def train_one_epoch(self,config,models, criterions, data_loader, optimizer, epoch, mixup_fn=None, lr_scheduler=None,amp_autocast=suppress,loss_scaler=None,model_ema=None,logger=None,**kwargs):
        models['main'].train()
        if config.EMPTY_CACHE:
            torch.cuda.empty_cache()
        optimizer.zero_grad()
        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        num_steps = len(data_loader)
        is_timm_loader = config.DATA.DATALOADER_NAME.startswith('timm')

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        norm_meter = AverageMeter()
        scaler_meter = AverageMeter()
        reset_meter(self.engine.train_metrics)
        
        # do something before the training
        self.engine.update_before_train(config,epoch=epoch,models=models,dataloader=data_loader)

        loss_rec = np.array([])
        start = time.time()
        end = time.time()
        last_idx = len(data_loader) - 1
        
        for idx, (samples, targets) in enumerate(data_loader):
            last_batch = idx == last_idx
            # timm dataloader prefetcher will do this
            if not is_timm_loader or not config.DATA.TIMM_PREFETCHER:
                if type(samples) in (tuple,list):
                    for _i in range(len(samples)):
                        samples[_i] = samples[_i].cuda(non_blocking=config.DATA.PIN_MEMORY)
                else:
                    samples = samples.cuda(non_blocking=config.DATA.PIN_MEMORY)
                        
                targets = targets.cuda(non_blocking=config.DATA.PIN_MEMORY)
            # timm dataloader prefetcher will do this
            if mixup_fn is not None and not is_timm_loader:
                samples, targets = mixup_fn(samples, targets)

            with amp_autocast():
                loss,metrics_values,output = self.engine.cal_loss_func(config,models,idx,samples,targets,epoch,num_steps,criterions)
                if isinstance(output, (tuple, list)):
                    predictions = output[0]

                loss = loss / config.TRAIN.ACCUMULATION_STEPS

            if loss_scaler is not None:
                grad_norm = loss_scaler(loss,optimizer,
                    clip_grad=None if config.TRAIN.CLIP_GRAD == 0 else config.TRAIN.CLIP_GRAD, clip_mode=config.TRAIN.CLIP_MODE,
                    parameters=model_parameters(models['main'], exclude_head='agc' in config.TRAIN.CLIP_MODE>0), 
                    update_grad = (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
                    change_grad_func=self.engine.change_grad_func,
                    config=config,
                    epoch=epoch,
                    models=models,
                    idx=idx)
                loss_scale_value = loss_scaler.state_dict()["scale"]
            else:
                loss.backward(create_graph=second_order)
                if config.TRAIN.CLIP_GRAD > 0:
                    dispatch_clip_grad(
                        model_parameters(models['main'], exclude_head='agc' in config.TRAIN.CLIP_MODE>0),
                        value=config.TRAIN.CLIP_GRAD, mode=config.TRAIN.CLIP_MODE)

                self.engine.change_grad_func(config,epoch,models,idx)

                grad_norm = ampscaler_get_grad_norm(model_parameters(models['main'], exclude_head='agc' in config.TRAIN.CLIP_MODE>0))
                loss_scale_value = 0.

            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if loss_scaler is None:
                    optimizer.step()

                optimizer.zero_grad()

                if model_ema is not None:
                    model_ema.update(models['main'],epoch*config.DATA.LEN_DATALOADER_TRAIN+idx)
                
                lr_scheduler.step((epoch * num_steps + idx)// config.TRAIN.ACCUMULATION_STEPS)

                # 每个iter更新
                self.engine.update_per_iter(config,epoch,idx,models,output=output)

            torch.cuda.synchronize()
            if config.DISTRIBUTED:
                    if grad_norm is not None:
                        reduced_norm = reduce_tensor(grad_norm,config.WORLD_SIZE)
                        norm_meter.update(reduced_norm)
                    reduced_loss = reduce_tensor(loss.data,config.WORLD_SIZE)
                    reduced_scalar = loss_scale_value
                    loss_meter.update(reduced_loss.item(), targets.size(0))
                    scaler_meter.update(reduced_scalar)
                    update_metrics(config,self.engine.train_metrics,metrics_values,distributed=True)
            else:
                if grad_norm is not None:
                    norm_meter.update(grad_norm)
                loss_meter.update(loss.item(), targets.size(0))
                scaler_meter.update(loss_scale_value)
                update_metrics(config,self.engine.train_metrics,metrics_values)

            batch_time.update(time.time() - end)
            np.append(loss_rec,loss_meter.avg)
            end = time.time()

            if last_batch or idx % config.PRINT_FREQ == 0:
                #lr = optimizer.param_groups[0]['lr']
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)

                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                    f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
                # log per iter
                wandb_dic_iter = log_meter(self.engine.train_metrics,self.engine.train_metrics_iter_log,logger)
                # wandb log per iter
                if config.LOG_WANDB and has_wandb and config.LOCAL_RANK == 0:
                    rowd = OrderedDict([('iter/loss',loss_meter.val),('iter/grad_norm',norm_meter.val),('iter/lr',lr)])
                    rowd.update(wandb_dic_iter)
                    
                    if config.TEST.K_FOLD_VAL_ENABLE:
                        rowd = OrderedDict([ (str(config.TEST.K_FOLD_VAL_NOW)+'-fold/'+k,v) for k, v in rowd.items()])
                        print(rowd)

                    wandb.log(rowd,step=epoch * num_steps + idx)
        #每一轮更新一次
        train_metrics_epoch_add = self.engine.update_per_epoch(config,epoch,models=models,dataloader=data_loader)
        if train_metrics_epoch_add is not None:
           self.engine.train_metrics.update(train_metrics_epoch_add) 

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        epoch_time = time.time() - start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
        # log per epoch
        wandb_dic_epoch = log_meter(self.engine.train_metrics,self.engine.train_metrics_epoch_log,logger)

        if config.EMPTY_CACHE:
            torch.cuda.empty_cache()
        return_train_metrics = OrderedDict([('loss', loss_meter.avg),('grad_norm',norm_meter.avg),('loss_scale',scaler_meter.avg)])
        return_train_metrics.update(wandb_dic_epoch)
        return loss,return_train_metrics

    def predict(config, data_loader, model,amp_autocast=suppress,logger=None):
        model.eval()
        torch.cuda.empty_cache()

        batch_time = AverageMeter()
        
        save_pred = np.array([])
        save_label = np.array([])

        end = time.time()
        last_idx = len(data_loader) - 1
        
        with torch.no_grad():
            for idx, (images, targets) in enumerate(data_loader):
                last_batch = idx == last_idx
                # timm dataloader prefetcher will do this
                if not config.DATA.TIMM or not config.DATA.TIMM_PREFETCHER:
                    images = images.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)

                #if config.EVAL_MODE:
                targets_bin = targets.clone()

                if 'cqu_bpdd' in config.DATA.DATASET:
                    targets_bin[targets==config.DATA.DATA_NOR_INDEX] = 0
                    targets_bin[targets!=config.DATA.DATA_NOR_INDEX] = 1
                # compute output
                with amp_autocast():
                    output = model(images)
                if isinstance(output, (tuple, list)):
                    index = 0 if config.THUMB_MODE or config.PICT.NOT_INST_TEST else 1

                    output = output[index]

                output_soft = torch.nn.functional.softmax(output,dim=-1)
        
                save_pred = np.append(save_pred,output_soft.cpu().numpy())
                save_label = np.append(save_label,targets.cpu().numpy())
                
                torch.cuda.synchronize()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if last_batch or idx % config.PRINT_FREQ == 0:
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    logger.info(
                        f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')
                
            save_pred = save_pred.reshape(-1,config.MODEL.NUM_CLASSES)

        torch.cuda.empty_cache()

        return save_pred,save_label
   
    @torch.no_grad()
    def validate(self, config, data_loader, models,save_pre=False,amp_autocast=suppress,criterions=None,logger=None,is_test=False,**kwargs):
        for _key in models.keys():
            models[_key].eval()

        if config.EMPTY_CACHE:
            torch.cuda.empty_cache()
        is_timm_loader = config.DATA.DATALOADER_NAME.startswith('timm')

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        reset_meter(self.engine.test_metrics)

        save_pred = np.array([])
        save_label = np.array([])

        end = time.time()
        last_idx = len(data_loader) - 1
        
        for idx, (samples, targets) in enumerate(data_loader):
            last_batch = idx == last_idx
            # timm dataloader prefetcher will do this
            if not is_timm_loader or not config.DATA.TIMM_PREFETCHER:
                if type(samples) in (tuple,list):
                    for _i in range(len(samples)):
                        samples[_i] = samples[_i].cuda(non_blocking=config.DATA.PIN_MEMORY)
                else:
                    samples = samples.cuda(non_blocking=config.DATA.PIN_MEMORY)
                targets = targets.cuda(non_blocking=True)

            with amp_autocast():
                loss,pred,label,metrics_values,others = self.engine.measure_per_iter(config,models,samples,targets,criterions,is_test=is_test)

            save_pred = np.append(save_pred,pred)
            save_label = np.append(save_label,label)

            if config.DISTRIBUTED:
                loss = reduce_tensor(loss,config.WORLD_SIZE)
            loss_meter.update(loss.item(), targets.size(0))
            update_metrics(config,self.engine.test_metrics,metrics_values,config.DISTRIBUTED)

            torch.cuda.synchronize()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if last_batch or idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Mem {memory_used:.0f}MB')
                log_meter(self.engine.test_metrics,self.engine.test_metrics_iter_log,logger)
            
            save_pred = save_pred.reshape(-1,config.MODEL.NUM_CLASSES)

        metrics_values_epoch,others_epoch = self.engine.measure_per_epoch(config,others=others,label=save_label,pred=save_pred,is_test=is_test)
        update_metrics(config,self.engine.test_metrics,metrics_values_epoch)
        log_meter(metrics_values_epoch,self.engine.test_metrics_epoch_log,logger)

        metrics = OrderedDict([(key,value.avg) if isinstance(value,AverageMeter) else (key,value) for key,value in self.engine.test_metrics.items()])
        if config.EMPTY_CACHE:
            torch.cuda.empty_cache()
        if save_pre:
            return loss_meter.avg, metrics, save_pred, save_label
        else:    
            return loss_meter.avg, metrics