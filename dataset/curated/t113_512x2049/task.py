import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 113
M, D, DT = 512, 2049, torch.bfloat16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = torch.relu(x)
        x = torch.softmax(x, dim=-1)
        x = torch.relu(x)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
