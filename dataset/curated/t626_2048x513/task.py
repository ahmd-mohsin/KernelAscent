import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 626
M, D, DT = 2048, 513, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x * 1.0029
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        x = torch.softmax(x, dim=-1)
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
