import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 200001
S, D, DT = 512, 512, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_cache(self, x):
        dev = x.device
        s = x.shape[0]
        cache = getattr(self, "_cache", None)
        if cache is not None and cache[0] == dev and cache[1] == s and cache[2].dtype == self.Wq.dtype:
            return cache[2], cache[3], cache[4]
        d = self.Wq.shape[0]
        # Fused QKV weight (column-concat => identical per-column GEMM results)
        Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
        # RoPE tables (float32, matching reference math)
        half = d // 2
        pos = torch.arange(s, device=dev, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, half, device=dev, dtype=torch.float32)
            * (-math.log(10000.0) / max(half, 1))
        )
        ang = pos * freq
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        self._cache = (dev, s, Wqkv, cos, sin)
        return Wqkv, cos, sin

    def forward(self, x):
        d = self.Wq.shape[0]
        half = d // 2
        Wqkv, cos, sin = self._get_cache(x)

        # Single fused projection GEMM
        qkv = x @ Wqkv  # (s, 3d)
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        dt = x.dtype
        # RoPE in float32 (matches reference numerics), applied to q and k together
        q1 = q[:, :half].float(); q2 = q[:, half:2 * half].float()
        k1 = k[:, :half].float(); k2 = k[:, half:2 * half].float()

        qr = torch.empty_like(q)
        kr = torch.empty_like(k)
        torch.addcmul(q1 * cos, q2, sin, value=-1.0, out=None)  # placeholder removed below
        qr[:, :half] = (q1 * cos - q2 * sin).to(dt)
        qr[:, half:2 * half] = (q1 * sin + q2 * cos).to(dt)
        if 2 * half < d:
            qr[:, 2 * half:] = q[:, 2 * half:]
        kr[:, :half] = (k1 * cos - k2 * sin).to(dt)
        kr[:, half:2 * half] = (k1 * sin + k2 * cos).to(dt)
        if 2 * half < d:
            kr[:, 2 * half:] = k[:, 2 * half:]

        # Fused causal attention (FlashAttention on A100)
        out = F.scaled_dot_product_attention(
            qr.unsqueeze(0).unsqueeze(0),
            kr.unsqueeze(0).unsqueeze(0),
            v.contiguous().unsqueeze(0).unsqueeze(0),
            is_causal=True,
            scale=1.0 / math.sqrt(d),
        )
        return out.squeeze(0).squeeze(0)
