import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 314
M, D, DT = 1024, 4097, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = F.gelu(x)
        x = torch.relu(x)
        x = F.gelu(x)
        x = x * 1.2863
        x = torch.softmax(x, dim=-1)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
