import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 200017
S, D, DT = 1024, 1024, torch.float16


@triton.jit
def _rope_kernel(ptr, cos_ptr, sin_ptr, stride,
                 HALF: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    which = tl.program_id(1)  # 0 -> q, 1 -> k
    offs = tl.arange(0, BLOCK)
    mask = offs < HALF
    base = ptr + row * stride + which * (2 * HALF)
    t1 = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    t2 = tl.load(base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + row * HALF + offs, mask=mask, other=0.0)
    s = tl.load(sin_ptr + row * HALF + offs, mask=mask, other=0.0)
    o1 = t1 * c - t2 * s
    o2 = t1 * s + t2 * c
    tl.store(base + offs, o1.to(tl.float16), mask=mask)
    tl.store(base + HALF + offs, o2.to(tl.float16), mask=mask)


@triton.jit
def _causal_softmax_kernel(x_ptr, out_ptr, n_cols, scale,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    in_row = offs < n_cols
    causal = offs <= row
    x = tl.load(x_ptr + row * n_cols + offs, mask=in_row & causal,
                other=float('-inf')).to(tl.float32)
    x = x * scale
    x = tl.where(causal, x, float('-inf'))
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(causal & in_row, e, 0.0)
    denom = tl.sum(e, 0)
    p = e / denom
    tl.store(out_ptr + row * n_cols + offs, p.to(tl.float16), mask=in_row)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def _prep(self, device, seq_len, dim):
        if getattr(self, '_Wqkv', None) is None or self._Wqkv.device != device:
            self._Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).contiguous()
        if (getattr(self, '_cos', None) is None or self._cos.device != device
                or self._cos.shape[0] != seq_len or self._cos.shape[1] != dim // 2):
            half = dim // 2
            pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
            freq = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32)
                             * (-math.log(10000.0) / max(half, 1)))
            ang = pos * freq
            self._cos = torch.cos(ang).contiguous()
            self._sin = torch.sin(ang).contiguous()

    def forward(self, x):
        device = x.device
        seq_len, dim = x.shape
        self._prep(device, seq_len, dim)
        half = dim // 2

        # Fused QKV projection (single GEMM)
        qkv = x @ self._Wqkv  # (S, 3D), contiguous

        # Fused RoPE on q and k (in-place)
        BLOCK_H = triton.next_power_of_2(half)
        _rope_kernel[(seq_len, 2)](
            qkv, self._cos, self._sin, qkv.stride(0),
            HALF=half, BLOCK=BLOCK_H,
        )

        q = qkv[:, :dim]
        k = qkv[:, dim:2 * dim]
        v = qkv[:, 2 * dim:]

        # Attention scores (GEMM), then fused scale + causal mask + softmax
        scores = q @ k.transpose(-1, -2)  # (S, S) fp16
        a = torch.empty_like(scores)
        scale = 1.0 / math.sqrt(dim)
        BLOCK_S = triton.next_power_of_2(seq_len)
        _causal_softmax_kernel[(seq_len,)](
            scores, a, seq_len, scale, BLOCK=BLOCK_S,
        )

        return a @ v
