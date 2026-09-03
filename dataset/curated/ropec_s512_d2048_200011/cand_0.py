import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200011
S, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _rope_kernel(ptr, cos_ptr, sin_ptr, half, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half
    base = ptr + row * stride
    t1 = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    t2 = tl.load(base + half + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + row * half + offs, mask=mask, other=0.0)
    s = tl.load(sin_ptr + row * half + offs, mask=mask, other=0.0)
    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c
    tl.store(base + offs, o1.to(tl.bfloat16), mask=mask)
    tl.store(base + half + offs, o2.to(tl.bfloat16), mask=mask)


@triton.jit
def _causal_softmax_kernel(s_ptr, o_ptr, n, scale,
                           s_stride, o_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    causal = offs <= row
    x = tl.load(s_ptr + row * s_stride + offs, mask=causal,
                other=float('-inf')).to(tl.float32) * scale
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(causal, e, 0.0)
    d = tl.sum(e, 0)
    y = e / d
    tl.store(o_ptr + row * o_stride + offs, y.to(tl.bfloat16), mask=offs < n)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _get_cache(self, x):
        s, d = x.shape
        key = (s, d, x.device)
        cache = getattr(self, "_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2], cache[3]
        Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(x.device)
        half = d // 2
        pos = torch.arange(s, device=x.device, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(torch.arange(0, half, device=x.device, dtype=torch.float32)
                         * (-math.log(10000.0) / max(half, 1)))
        ang = pos * freq
        cos = torch.cos(ang).contiguous()
        sin = torch.sin(ang).contiguous()
        self._cache = (key, Wqkv, cos, sin)
        return Wqkv, cos, sin

    def forward(self, x):
        s, d = x.shape
        half = d // 2
        Wqkv, cos, sin = self._get_cache(x)

        # fused QKV projection (one GEMM instead of three)
        qkv = x @ Wqkv                      # (s, 3d)
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # in-place RoPE on q and k (fp32 math, bf16 storage)
        BLOCK_H = triton.next_power_of_2(half)
        _rope_kernel[(s,)](q, cos, sin, half, qkv.stride(0), BLOCK=BLOCK_H)
        _rope_kernel[(s,)](k, cos, sin, half, qkv.stride(0), BLOCK=BLOCK_H)

        # attention scores
        scores = q @ k.transpose(-1, -2)    # (s, s) bf16 GEMM

        # fused scale + causal mask + softmax
        a = torch.empty((s, s), device=x.device, dtype=x.dtype)
        BLOCK_S = triton.next_power_of_2(s)
        _causal_softmax_kernel[(s,)](
            scores, a, s, 1.0 / math.sqrt(d),
            scores.stride(0), a.stride(0), BLOCK=BLOCK_S,
        )

        return a @ v
