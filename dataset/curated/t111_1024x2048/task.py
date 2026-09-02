import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 111
M, D, DT = 1024, 2048, torch.bfloat16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W2 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = F.gelu(x)
        x = torch.softmax(x, dim=-1)
        x = x @ self.W2
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        x = x * 1.4031
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
