import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 877
M, D, DT = 1024, 4096, torch.bfloat16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = torch.softmax(x, dim=-1)
        x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
        x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
        x = torch.relu(x)
        return x

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
