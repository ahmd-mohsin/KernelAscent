import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 100024
S, D, DT = 2048, 512, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_fused_w(self, device, dtype):
        w = getattr(self, "_w_qkv", None)
        if w is None or w.device != device or w.dtype != dtype:
            w = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(device=device, dtype=dtype).contiguous()
            self._w_qkv = w
        return w

    def forward(self, x):
        # Fused QKV projection: one big GEMM instead of three
        w = self._get_fused_w(x.device, x.dtype)
        qkv = x @ w
        d = self.Wq.shape[1]
        q = qkv[..., :d]
        k = qkv[..., d:2 * d]
        v = qkv[..., 2 * d:]

        # Fused attention (memory-efficient / flash kernel) — avoids
        # materializing the SxS score matrix and fuses softmax.
        try:
            q4 = q.unsqueeze(0).unsqueeze(0)
            k4 = k.unsqueeze(0).unsqueeze(0)
            v4 = v.unsqueeze(0).unsqueeze(0)
            out = F.scaled_dot_product_attention(q4, k4, v4)
            return out.squeeze(0).squeeze(0)
        except Exception:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            a = torch.softmax(scores, dim=-1)
            return a @ v
