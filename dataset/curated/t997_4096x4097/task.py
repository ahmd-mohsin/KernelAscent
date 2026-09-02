import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 997
M, D, DT = 4096, 4097, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
        x = x + self.b1
        x = torch.softmax(x, dim=-1)
        x = x + self.b3
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
