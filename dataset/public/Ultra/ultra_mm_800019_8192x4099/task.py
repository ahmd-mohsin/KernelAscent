import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 800019
M, D, DT = 8192, 4099, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4099, 1024, generator=g) / math.sqrt(4099)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = F.gelu(x)
        x = x @ self.W2
        x = torch.softmax(x, dim=-1)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
