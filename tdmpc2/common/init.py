import torch.nn as nn
import torch

DTYPE = torch.float
DEVICE = 'cuda:0'


def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i,p in enumerate(m):
            if p.dim() == 3: # Linear
                nn.init.trunc_normal_(p, std=0.02) # Weight
                nn.init.constant_(m[i+1], 0) # Bias


def zero_(params):
    """Initialize parameters to zero."""
    for p in params:
        p.data.fill_(0)


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x

def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
    return torch.tensor(x, dtype=dtype, device=device)