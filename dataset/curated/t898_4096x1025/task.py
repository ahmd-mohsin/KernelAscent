import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 898
M, D, DT = 4096, 1025, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = torch.softmax(x, dim=-1)
        x = torch.relu(x)
        x = torch.relu(x)
        x = F.gelu(x)
        x = F.layer_norm(x, (x.shape[-1],), self.ln5_g, self.ln5_b)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
