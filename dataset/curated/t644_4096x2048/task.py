import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 644
M, D, DT = 4096, 2048, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = torch.softmax(x, dim=-1)
        x = x + self.b1
        x = F.gelu(x)
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
