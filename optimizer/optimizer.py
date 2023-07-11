#from torch import optim as optim
from timm.optim import create_optimizer_v2
from .ranger import RangerLars
import torch.nn as nn

def build_optimizer(config, model):
    """
    Use the timm optimizer
    """
    param,weight_decay = get_param(config,model)
    if config.TRAIN.OPTIMIZER.NAME.lower() == 'rangerlars':
        optimizer = RangerLars(param,lr=config.TRAIN.BASE_LR, weight_decay=weight_decay)
    else:
        optimizer = create_optimizer_v2(param,config.TRAIN.OPTIMIZER.NAME,config.TRAIN.BASE_LR,weight_decay,config.TRAIN.OPTIMIZER.MOMENTUM,eps=config.TRAIN.OPTIMIZER.EPS)
    return optimizer

# copy from timm and add the names
def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    decay_names = []
    no_decay_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # filter bias and bn
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decay.append(param)
            decay_names.append(name)

    return [
        {'params': no_decay, 'weight_decay': 0., 'names': no_decay_names},

        {'params': decay, 'weight_decay': weight_decay, 'names': decay_names}]

def get_param(
        config,
        model: nn.Module
):
    no_weight_decay = {}
    if hasattr(model, 'no_weight_decay'):
        no_weight_decay = model.no_weight_decay()

    param = param_groups_weight_decay(model,config.TRAIN.WEIGHT_DECAY,no_weight_decay)

    if config.TRAIN.OPTIMIZER.PARAM_GROUPS_FUNC is None:
        pass
    elif config.TRAIN.OPTIMIZER.PARAM_GROUPS_FUNC == 'simsiam':
        param = simsiam_param_groping(config,param)
    else:
        raise NotImplementedError
    return param , 0.

def simsiam_param_groping(config,param_groups):
    _param_groups = []
    predictor_prefix = ('module.predictor', 'predictor')
    for param_group in param_groups:
        predictor_params = []
        backbone_params = []

        for _name,_param in zip(param_group['names'],param_group['params']):
            if _name.startswith(predictor_prefix):
                predictor_params.append(_param)
            else:
                backbone_params.append(_param)

        if len(predictor_params) != 0:
            _param_groups.append({'params':predictor_params,'weight_decay':param_group['weight_decay'],'lr':config.TRAIN.BASE_LR, 'name':'predictor'})
        if len(backbone_params) != 0:
            _param_groups.append({'params':backbone_params,'weight_decay':param_group['weight_decay'],'lr':config.TRAIN.BASE_LR, 'name':'backbone'})
    return _param_groups