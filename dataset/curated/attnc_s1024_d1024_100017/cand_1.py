import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100017
S, D, DT = 1024, 1024, torch.float16


@triton.jit
def _causal_softmax_kernel(
    S_ptr,          # [n, n] fp16 raw scores (q @ k^T, unscaled)
    O_ptr,          # [n, n] fp16 output probabilities
    n,              # number of columns (== rows)
    scale,          # 1/sqrt(d)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_bounds = cols < n

    x = tl.load(S_ptr + row * n + cols, mask=in_bounds, other=0.0).to(tl.float32)
    x = x * scale

    # causal mask: positions j > i get -inf
    keep = in_bounds & (cols <= row)
    x = tl.where(keep, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    tl.store(O_ptr + row * n + cols, p.to(O_ptr.dtype.element_ty), mask=in_bounds)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.Wq = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wk = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)
        self.Wv = nn.Parameter((torch.randn(D, D, generator=g) / math.sqrt(D)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Lazily build the fused QKV weight once (single big GEMM instead of 3)
        Wqkv = getattr(self, "_Wqkv", None)
        if Wqkv is None or Wqkv.device != x.device or Wqkv.dtype != x.dtype:
            Wqkv = torch.cat([self.Wq, self.Wk, self.Wv], dim=1).to(x.device, x.dtype).contiguous()
            self._Wqkv = Wqkv

        d = x.shape[-1]
        x2 = x.contiguous()

        # Fused projection: one GEMM for Q, K, V
        qkv = x2 @ Wqkv
        q, k, v = qkv.split(d, dim=-1)

        # Raw attention scores via tensor-core GEMM (scale fused into softmax kernel;
        # division by sqrt(1024)=32 is an exact power-of-two scaling, so applying it
        # in fp32 inside the kernel is numerically identical)
        scores = q @ k.transpose(-1, -2)
        scores = scores.contiguous()

        n = scores.shape[-1]
        probs = torch.empty_like(scores)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _causal_softmax_kernel[(scores.shape[0],)](
            scores, probs, n, 1.0 / math.sqrt(d),
            BLOCK=BLOCK, num_warps=num_warps,
        )

        return probs @ v
