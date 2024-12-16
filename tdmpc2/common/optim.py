import torch
import math
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from typing import Union, Iterable, Tuple, Callable, List
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random

def configure_weight_decay(model: nn.Module, weight_decay: float) -> List:
    """
    Overview:
        Separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layer-norm or embedding weights).
    Arguments:
        - model (:obj:`nn.Module`): The given PyTorch model.
        - weight_decay (:obj:`float`): Weight decay value for optimizer.
    Returns:
        - optim groups (:obj:`List`): The parameter groups to be set in the latter optimizer.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            # Because named_modules and named_parameters are recursive
            # we will see the same tensors p many times. But doing it this way
            # allows us to know which parent module any tensor p belongs to.
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            else:
                decay.add(fpn)

    decay = decay - no_decay
    # for k in decay:
    #     print(k)
    # print('\n\n\n\nnot decay:\n\n\n')
    # for k in no_decay:
    #     print(k)
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    union_params = decay | no_decay
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0
        },
    ]

    return optim_groups