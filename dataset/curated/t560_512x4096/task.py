import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 560
M, D, DT = 512, 4096, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = F.gelu(x)
        x = x * 1.2123
        x = torch.relu(x)
        x = F.gelu(x)
        x = x + self.b4
        x = torch.softmax(x, dim=-1)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
