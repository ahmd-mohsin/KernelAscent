import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 368
M, D, DT = 4096, 513, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = torch.relu(x)
        x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
        x = x + self.b2
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        x = torch.relu(x)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
