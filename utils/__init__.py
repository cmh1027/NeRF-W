import torch
# optimizer
from torch.optim import SGD, Adam
import torch_optimizer as optim
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .warmup_scheduler import GradualWarmupScheduler

from .visualization import *
from . import barf

def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else: # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters

def get_optimizer(type, lr, models):
    eps = 1e-8
    parameters = get_parameters(models)
    if type == 'sgd':
        optimizer = SGD(parameters, lr=lr, 
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif type == 'adam':
        optimizer = Adam(parameters, lr=lr, eps=eps, 
                        #  weight_decay=hparams.weight_decay)
        )
    elif type == 'radam':
        optimizer = optim.RAdam(parameters, lr=lr, eps=eps, 
                                weight_decay=hparams.weight_decay)
    elif type == 'ranger':
        optimizer = optim.Ranger(parameters, lr=lr, eps=eps, 
                                 weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def get_scheduler(type, lr, lr_end, max_step, optimizer):
    eps = 1e-8
    scheduler_module = getattr(torch.optim.lr_scheduler,type)
    if lr_end:
        assert(type=="ExponentialLR")
        gamma = (lr_end/lr)**(1./max_step)
    scheduler = scheduler_module(optimizer, gamma=gamma)
    # if hparams.lr_scheduler == 'steplr':
    #     scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, 
    #                             gamma=hparams.decay_gamma)
    # elif hparams.lr_scheduler == 'cosine':
    #     scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_steps, eta_min=eps)
    # elif hparams.lr_scheduler == 'poly':
    #     scheduler = LambdaLR(optimizer, 
    #                          lambda epoch: (1-epoch/hparams.num_epochs)**hparams.poly_exp)
    # else:
    #     raise ValueError('scheduler not recognized!')

    # if hparams.warmup_epochs > 0 and hparams['optimizer.type'] not in ['radam', 'ranger']:
    #     scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier, 
    #                                        total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)
