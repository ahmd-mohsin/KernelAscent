import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 789
M, D, DT = 4096, 512, torch.bfloat16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        x = x * 1.3032
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.softmax(x, dim=-1)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
