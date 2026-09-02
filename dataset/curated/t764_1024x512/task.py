import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 764
M, D, DT = 1024, 512, torch.bfloat16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = torch.softmax(x, dim=-1)
        x = x * 1.3875
        x = F.gelu(x)
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
