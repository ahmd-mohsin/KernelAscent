import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200013
S, D, DT = 1024, 512, torch.float16


def _rope(t):
    S_, E = t.shape
    half = E // 2
    pos = torch.arange(S_, device=t.device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
    ang = pos * freq
    cos, sin = torch.cos(ang), torch.sin(ang)
    t1 = t[..., :half].float(); t2 = t[..., half:half * 2].float()
    out = t.float().clone()
    out[..., :half] = t1 * cos - t2 * sin
    out[..., half:half * 2] = t1 * sin + t2 * cos
    return out.to(t.dtype)


@triton.jit
def _rope_qk_kernel(qkv_ptr, cos_ptr, sin_ptr,
                    D: tl.constexpr, HALF: tl.constexpr, STRIDE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, HALF)
    c = tl.load(cos_ptr + row * HALF + offs)
    s = tl.load(sin_ptr + row * HALF + offs)
    base = qkv_ptr + row * STRIDE
    # rotate q (first D columns)
    q1 = tl.load(base + offs).to(tl.float32)
    q2 = tl.load(base + HALF + offs).to(tl.float32)
    tl.store(base + offs, (q1 * c - q2 * s).to(tl.float16))
    tl.store(base + HALF + offs, (q1 * s + q2 * c).to(tl.float16))
    # rotate k (next D columns)
    k1 = tl.load(base + D + offs).to(tl.float32)
    k2 = tl.load(base + D + HALF + offs).to(tl.float32)
    tl.store(base + D + offs, (k1 * c - k2 * s).to(tl.float16))
    tl.store(base + D + HALF + offs, (k1 * s + k2 * c).to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_cache(self, device, seq_len, half):
        cache = getattr(self, "_cache", None)
        if (cache is None or cache[0] != device or cache[1] != seq_len or cache[2] != half):
            pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32)
                             * (-math.log(10000.0) / max(half, 1)))
            ang = pos * freq
            cos = torch.cos(ang).contiguous()
            sin = torch.sin(ang).contiguous()
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
            self._cache = (device, seq_len, half, cos, sin, Wqkv)
            cache = self._cache
        return cache[3], cache[4], cache[5]

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            q = _rope(x @ self.Wq); k = _rope(x @ self.Wk); v = x @ self.Wv
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            scores = scores + torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
            a = torch.softmax(scores, dim=-1)
            return a @ v

        seq_len, d = x.shape
        half = d // 2
        cos, sin, Wqkv = self._get_cache(x.device, seq_len, half)

        # fused QKV projection: one GEMM instead of three
        qkv = torch.mm(x, Wqkv)  # (S, 3D), contiguous

        # fused RoPE on q and k in one kernel launch
        _rope_qk_kernel[(seq_len,)](qkv, cos, sin, d, half, 3 * d, num_warps=4)

        q = qkv[:, :d].unsqueeze(0).unsqueeze(0)
        k = qkv[:, d:2 * d].unsqueeze(0).unsqueeze(0)
        v = qkv[:, 2 * d:].unsqueeze(0).unsqueeze(0)

        # fused causal attention (FlashAttention) with fp32 softmax accumulation
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.squeeze(0).squeeze(0)
