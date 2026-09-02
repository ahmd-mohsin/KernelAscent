import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 400002
S, D, N, E, DT = 512, 1024, 2048, 4, torch.float16

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wr = nn.Parameter((torch.randn(D, E, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.We = nn.Parameter((torch.randn(E, D, N, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        gate = torch.softmax(x @ self.Wr, dim=-1)
        outs = torch.stack([x @ self.We[e] for e in range(E)], dim=0)
        y = (gate.transpose(0, 1).unsqueeze(-1) * outs).sum(0)
        return F.gelu(y)

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
