import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 786
M, D, DT = 2048, 512, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x + self.b0
        x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
        x = x * 1.4996
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
