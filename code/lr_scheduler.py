import torch
from torch.optim.lr_scheduler import MultiStepLR
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
import math
from typing import Dict, Any

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch) if int(config.TRAIN.WARMUP_STEPS) == -1 else int(config.TRAIN.WARMUP_STEPS) // config.TRAIN.ACCUMULATION_STEPS
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME is None:
        pass
    elif config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            cycle_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'flat_cosine':
        lr_scheduler = flat_cosine_schedule(
            optimizer,
            num_flat_steps = float(num_steps*config.TRAIN.LR_SCHEDULER.DECAY_STEPS_RATIO)-1,
            num_training_steps = num_steps
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'multi_step':
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones = config.TRAIN.LR_SCHEDULER.MULTISTEPS, 
            gamma = config.TRAIN.LR_SCHEDULER.GAMMA
        )

    else: 
        raise NotImplementedError
    return SchedulerWrapper(lr_scheduler,config)

def flat_cosine_schedule(optimizer: torch.optim.Optimizer, num_flat_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    def lr_lambda(current_step):
        if current_step < num_flat_steps:
            return 1.0
        progress = float(current_step - num_flat_steps) / float(max(1, num_training_steps - num_flat_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

'''
For constant some param group in the lr scheduler and
Compatible with torch api and timm api 
'''
class SchedulerWrapper():
    def __init__(self,scheduler,config) -> None:
        self.scheduler = scheduler
        self.constant_lr_field = config.TRAIN.LR_SCHEDULER.CONSTANT_LR_FIELD if config.TRAIN.OPTIMIZER.PARAM_GROUPS_FUNC is not None else None
        self.base_lr = config.TRAIN.BASE_LR

    def state_dict(self) -> Dict[str, Any]:
        self.scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict)

    def step(self, num_steps: int, metric: float = None) -> None:
        # compatible with torch and timm
        if isinstance(self.scheduler,torch.optim.lr_scheduler._LRScheduler):
            self.scheduler.step(num_steps)
        elif isinstance(self.scheduler,Scheduler):
            self.scheduler.step_update(num_steps,metric)

        # constant the some param group
        if self.constant_lr_field is not None:
            for param_group in self.scheduler.optimizer.param_groups:

                if isinstance(self.constant_lr_field,(list,tuple)):
                    if param_group['name'].startswith(tuple(self.constant_lr_field)):
                        param_group['lr'] = self.base_lr
                else:
                    if param_group['name'].startswith(self.constant_lr_field):
                        param_group['lr'] = self.base_lr
                break