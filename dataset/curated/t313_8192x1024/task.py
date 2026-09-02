import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 313
M, D, DT = 8192, 1024, torch.bfloat16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = torch.relu(x)
        x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
        x = x @ self.W2
        x = F.gelu(x)
        x = x * 1.1733
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
