import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 474
M, D, DT = 512, 512, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = torch.relu(x)
        x = x * 1.0381
        x = x + self.b2
        x = F.gelu(x)
        x = torch.softmax(x, dim=-1)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
