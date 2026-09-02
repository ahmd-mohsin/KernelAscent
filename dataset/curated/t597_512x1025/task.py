import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 597
M, D, DT = 512, 1025, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = torch.softmax(x, dim=-1)
        x = F.gelu(x)
        x = x * 1.2534
        x = torch.softmax(x, dim=-1)
        x = torch.relu(x)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
