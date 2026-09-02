import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 200015
S, D, DT = 1024, 512, torch.bfloat16

def _rope(t):
    S, E = t.shape
    half = E // 2
    pos = torch.arange(S, device=t.device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
    ang = pos * freq
    cos, sin = torch.cos(ang), torch.sin(ang)
    t1 = t[..., :half].float(); t2 = t[..., half:half * 2].float()
    out = t.float().clone()
    out[..., :half] = t1 * cos - t2 * sin
    out[..., half:half * 2] = t1 * sin + t2 * cos
    return out.to(t.dtype)

class Model(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        q = _rope(x @ self.Wq); k = _rope(x @ self.Wk); v = x @ self.Wv
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
        a = torch.softmax(scores, dim=-1)
        return a @ v

def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(S, D, generator=g).to(DT)]
