import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 158
M, D, DT = 1024, 513, torch.bfloat16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x * 1.0026
        x = x + self.b1
        x = torch.softmax(x, dim=-1)
        x = x + self.b3
        x = x * 1.0435
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
