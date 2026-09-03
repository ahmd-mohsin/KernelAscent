import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200009
S, D, DT = 512, 2048, torch.float16


@triton.jit
def _rope_kernel(
    t_ptr, cos_ptr, sin_ptr, out_ptr,
    half: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < half

    t1 = tl.load(t_ptr + row * E + offs, mask=mask, other=0.0).to(tl.float32)
    t2 = tl.load(t_ptr + row * E + half + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + row * half + offs, mask=mask, other=0.0)
    s = tl.load(sin_ptr + row * half + offs, mask=mask, other=0.0)

    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c

    tl.store(out_ptr + row * E + offs, o1.to(tl.float16), mask=mask)
    tl.store(out_ptr + row * E + half + offs, o2.to(tl.float16), mask=mask)


@triton.jit
def _causal_softmax_kernel(
    s_ptr, out_ptr, scale,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(s_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    x = tl.where(offs <= row, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(out_ptr + row * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None
        self._cos = None
        self._sin = None
        self._cache_key = None

    def _prep_cache(self, x):
        key = (x.shape[0], x.device, x.dtype)
        if self._cache_key == key and self._Wqkv is not None:
            return
        self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
        s = x.shape[0]
        half = self.Wq.shape[1] // 2
        pos = torch.arange(s, device=x.device, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(
            torch.arange(0, half, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half, 1))
        )
        ang = pos * freq
        self._cos = torch.cos(ang).contiguous()
        self._sin = torch.sin(ang).contiguous()
        self._cache_key = key

    def forward(self, x):
        x = x.contiguous()
        self._prep_cache(x)
        s, d = x.shape
        half = d // 2

        # Single fused GEMM for Q, K, V projections
        qkv = x @ self._Wqkv
        q_raw = qkv[:, :d].contiguous()
        k_raw = qkv[:, d:2 * d].contiguous()
        v = qkv[:, 2 * d:].contiguous()

        # Fused RoPE (fp32 math, fp16 output) for q and k
        q = torch.empty_like(q_raw)
        k = torch.empty_like(k_raw)
        BLOCK = triton.next_power_of_2(half)
        _rope_kernel[(s,)](q_raw, self._cos, self._sin, q, half, d, BLOCK, num_warps=8)
        _rope_kernel[(s,)](k_raw, self._cos, self._sin, k, half, d, BLOCK, num_warps=8)

        # Attention scores (tensor-core GEMM), then fused scale + causal mask + softmax
        scores = q @ k.transpose(-1, -2)
        a = torch.empty_like(scores)
        BLOCK_N = triton.next_power_of_2(s)
        _causal_softmax_kernel[(s,)](
            scores, a, 1.0 / math.sqrt(d), s, BLOCK_N, num_warps=8
        )

        return a @ v
