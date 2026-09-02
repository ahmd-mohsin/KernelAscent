import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 701
M, D, DT = 2048, 4097, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x + self.b0
        x = torch.softmax(x, dim=-1)
        x = x * 1.2808
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
