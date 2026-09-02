import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 415
M, D, DT = 2048, 1024, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        x = x * 1.0319
        x = F.gelu(x)
        x = x * 1.3895
        x = torch.softmax(x, dim=-1)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
