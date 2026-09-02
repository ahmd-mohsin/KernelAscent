import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 298
M, D, DT = 1024, 1024, torch.bfloat16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = torch.softmax(x, dim=-1)
        x = torch.relu(x)
        x = F.gelu(x)
        x = x @ self.W4
        x = F.gelu(x)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
