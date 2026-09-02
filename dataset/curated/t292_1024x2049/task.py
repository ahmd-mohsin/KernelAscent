import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 292
M, D, DT = 1024, 2049, torch.bfloat16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = torch.relu(x)
        x = x * 1.464
        x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
