# coding=utf-8
"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging
import abc
import sys
from torch.optim.lr_scheduler import LambdaLR
logger = logging.getLogger(__name__)


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


# --------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------- #
# Parent of all LRSchedules here.
class _LRSchedule(ABC):
    warn_t_total = False

    def __init__(self, warmup=0.002, t_total=-1, **kw):
        super(_LRSchedule, self).__init__(**kw)
        if t_total < 0:
            logger.warning("t_total value of {} results in schedule not being applied".format(t_total))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        warmup = max(warmup, 0.)
        self.warmup = float(warmup)
        self.t_total = float(t_total)
        self.warned_for_t_total_at_progress = -1

    def get_lr(self, step, nowarn=False):
        if self.t_total < 0:
            return 1.
        progress = float(step) / self.t_total
        ret = self.get_lr_(progress)
        # warning for exceeding t_total (only active with warmup_linear
        if not nowarn and self.warn_t_total and progress > 1. and progress > self.warned_for_t_total_at_progress:
            logger.warning("Training beyond specified 't_total'. Learning rate multiplier set to {}. Please set 't_total' of {} correctly.".format(ret, self.__class__.__name__))
            self.warned_for_t_total_at_progress = progress
        # end warning
        return ret

    @abc.abstractmethod
    def get_lr_(self, progress):
        return 1.


# --------------------------------------------------------------------------------------- #
class ConstantLR(_LRSchedule):
    def get_lr_(self, progress):
        return 1.


class WarmupCosineSchedule(_LRSchedule):
    warn_t_total = True

    def __init__(self, warmup=0.002, t_total=-1, cycles=.5, **kw):
        super(WarmupCosineSchedule, self).__init__(warmup=warmup, t_total=t_total, **kw)
        self.cycles = cycles

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        else:
            progress = (progress - self.warmup) / (1 - self.warmup)   # progress after warmup
            return 0.5 * (1. + math.cos(math.pi * self.cycles * 2 * progress))


class WarmupConstantSchedule(_LRSchedule):
    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1.


class WarmupLinearSchedule(_LRSchedule):
    warn_t_total = True

    def get_lr_(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)


SCHEDULES = {
    None:ConstantLR,
    "none":ConstantLR,
    "warmup_cosine":WarmupCosineSchedule,
    "warmup_constant":WarmupConstantSchedule,
    "warmup_linear":WarmupLinearSchedule
}


# --------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------- #
# BertAdam
"""
Implements BERT version of Adam algorithm with weight decay fix.
Params:
    lr: learning rate
    warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
    t_total: total number of training steps for the learning rate schedule, -1  means constant learning rate of 1. (no warmup regardless of warmup setting). Default: -1
    schedule: schedule to use for the warmup (see above).Can be `'warmup_linear'`, `'warmup_constant'`, `'warmup_cosine'`,`'none'`,
        `None` or a `_LRSchedule` object (see below).If `None` or `'none'`, learning rate is always kept constant. Default : `'warmup_linear'`
    b1: Adams b1. Default: 0.9
    b2: Adams b2. Default: 0.999
    e: Adams epsilon. Default: 1e-6
    weight_decay: Weight decay. Default: 0.01
    max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
"""
class BertAdam(Optimizer):
    def __init__(self, params, lr=required, warmup=-1, warmup_steps=-1, t_total=-1, schedule="warmup_linear",
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0, **kwargs):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not isinstance(schedule, _LRSchedule) and schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))

        # initialize schedule object
        if not isinstance(schedule, _LRSchedule):
            schedule_type = SCHEDULES[schedule]
            schedule = schedule_type(warmup=warmup, t_total=t_total)
        else:
            if warmup != -1 or t_total != -1:
                logger.warning("warmup and t_total on the optimizer are ineffective when _LRSchedule object is provided"
                               " as schedule.Please specify custom warmup and t_total in _LRSchedule object.")

        defaults = dict(lr=lr, warmup_steps=warmup_steps, schedule=schedule, b1=b1, b2=b2, e=e, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                lr_scheduled = group['lr']
                lr_scheduled *= group['schedule'].get_lr(state['step'])
                lr.append(lr_scheduled)
        print("learning rate{}".format(lr))
        return lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                state['step'] += 1

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])
                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # if group['weight_decay'] > 0.0:
                #     update += group['weight_decay'] * p.data

                if group['warmup_steps'] > state['step']:
                    lr_scheduled = group['lr']
                else:
                    lr_scheduled = group['lr']
                    lr_scheduled *= group['schedule'].get_lr(state['step']-group['warmup_steps'])

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

        return loss


# --------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------- #
# LMBertAdam
class LMBertAdam(Optimizer):
    def __init__(self, params, lr=required, warmup=-1, warmup_steps=-1, t_total=-1, schedule="warmup_linear",
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0, **kwargs):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not isinstance(schedule, _LRSchedule) and schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))

        # initialize schedule object
        if not isinstance(schedule, _LRSchedule):
            schedule_type = SCHEDULES[schedule]
            schedule = schedule_type(warmup=warmup, t_total=t_total)
        else:
            if warmup != -1 or t_total != -1:
                logger.warning("warmup and t_total on the optimizer are ineffective when _LRSchedule object is provided"
                               " as schedule.Please specify custom warmup and t_total in _LRSchedule object.")

        defaults = dict(lr=lr, warmup_steps=warmup_steps, schedule=schedule, b1=b1, b2=b2, e=e, weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        super(LMBertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                lr_scheduled = group['lr']
                lr_scheduled *= group['schedule'].get_lr(state['step'])
                lr.append(lr_scheduled)
        print("learning rate{}".format(lr))
        return lr

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                state['step'] += 1

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                next_m_1 = next_m/(1.0 - beta1 ** state["step"])
                next_v_1 = next_v/(1.0 - beta2 ** state["step"])
                update = next_m_1 / (next_v_1.sqrt() + group['e'])
                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['warmup_steps'] > state['step']:
                    lr_scheduled = group['lr']
                else:
                    lr_scheduled = group['lr']
                    lr_scheduled *= group['schedule'].get_lr(state['step']-group['warmup_steps'])

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

        return loss
