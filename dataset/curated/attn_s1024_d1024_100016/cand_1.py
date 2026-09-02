import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100016
S, D, DT = 1024, 1024, torch.float16


@triton.jit
def _scaled_softmax_kernel(
    x_ptr, out_ptr,
    N, stride_x, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=float('-inf'))
    # Match reference: scores are scaled and rounded to fp16 before softmax
    x = (x.to(tl.float32) * scale).to(tl.float16).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(out_ptr + row * stride_o + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self._Wqkv = None  # lazily built fused projection weight

    def forward(self, x):
        d = self.Wq.shape[0]
        # Build (and cache) fused QKV projection: one big GEMM instead of three
        if (self._Wqkv is None
                or self._Wqkv.device != x.device
                or self._Wqkv.dtype != x.dtype):
            self._Wqkv = torch.cat(
                [self.Wq, self.Wk, self.Wv], dim=1
            ).to(device=x.device, dtype=x.dtype).contiguous()

        qkv = x @ self._Wqkv  # (S, 3D) single GEMM
        q = qkv[:, :d]
        k = qkv[:, d:2 * d]
        v = qkv[:, 2 * d:]

        # Attention scores via tensor-core GEMM
        scores = q @ k.transpose(-1, -2)  # (S, S) fp16

        Srows, N = scores.shape
        a = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _scaled_softmax_kernel[(Srows,)](
            scores, a,
            N, scores.stride(0), a.stride(0),
            1.0 / math.sqrt(d),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

        return a @ v
