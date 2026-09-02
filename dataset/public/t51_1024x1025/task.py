import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 51
M, D, DT = 1024, 1025, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        x = F.gelu(x)
        x = torch.softmax(x, dim=-1)
        x = torch.relu(x)
        x = torch.relu(x)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
